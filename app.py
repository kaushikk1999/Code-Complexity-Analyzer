"""Complexity Lab Streamlit application."""

from __future__ import annotations

import html
from datetime import datetime
from typing import List, Optional

import streamlit as st

from analyzer import StaticAnalysisResult, analyze_code
from benchmarking import (
    BenchmarkResult,
    ScalingBenchmarkResult,
    docker_available,
    profile_settings,
    run_benchmark,
    run_benchmark_in_docker,
    run_scaling_benchmark,
    should_run_auto_benchmark,
)
from interview import InterviewGrade, build_follow_up_questions, grade_interview_answer
from llm import (
    AlgorithmPlannerResult,
    enhance_with_gemini,
    generate_algorithm_optimization_plan,
    generate_optimized_code_with_gemini,
)
from optimization import (
    OptimizationPlan,
    OptimizationStep,
    OptimizedCodeCandidate,
    build_optimization_plan,
    validate_optimized_candidate,
)
from scoring import ScoreBreakdown, calculate_optimization_score
from utils.constants import (
    APP_NAME,
    APP_TAGLINE,
    DEFAULT_CODE,
    DEFAULT_INPUT,
    DEFAULT_REPEAT_COUNT,
    DEFAULT_TIMEOUT_SECONDS,
    HISTORY_LIMIT,
    MAX_CODE_CHARS,
)
from utils.entrypoints import choose_entrypoint, discover_entrypoints
from utils.examples import EXAMPLES, get_example
from utils.history_store import load_recent_records, progress_summary, save_analysis_record
from utils.report_export import build_html_report, build_linkedin_summary, build_markdown_report
from utils.test_case_generator import generate_test_cases
from visualization.charts import (
    history_chart,
    memory_chart,
    runtime_chart,
    scaling_chart,
    score_breakdown_chart,
    score_gauge,
)
from visualization.styles import inject_global_styles
from visualization.ui_helpers import (
    complexity_badge,
    copyable_block,
    metric_card,
    render_empty_state,
    severity_pill,
)

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _rerun() -> None:
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
        return
    experimental_rerun = getattr(st, "experimental_rerun", None)
    if callable(experimental_rerun):
        experimental_rerun()


def _initialize_state() -> None:
    defaults = {
        "editor_code": DEFAULT_CODE,
        "benchmark_input": DEFAULT_INPUT,
        "entrypoint": "two_sum",
        "analysis": None,
        "benchmark": None,
        "scaling": None,
        "score": None,
        "plan": None,
        "gemini_text": None,
        "answer_grade": None,
        "algorithm_planner_question": "",
        "algorithm_planner_result": None,
        "algorithm_planner_submit_pending": False,
        "code_analyzer_plan_submit_pending": False,
        "gemini_api_key": "",
        "gemini_api_key_widget_version": 0,
        "pending_gemini_api_key": "",
        "gemini_prompt_decision": "",
        "gemini_key_requested": False,
        "benchmark_history": [],
        "practice_session": "Arrays",
        "static_only_mode": False,
        "docker_backend": False,
        "allow_top_level_benchmark": False,
        "interview_answer": "",
        "benchmark_input_entrypoint": "",
        "generated_test_cases": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _clear_outputs() -> None:
    st.session_state.analysis = None
    st.session_state.benchmark = None
    st.session_state.scaling = None
    st.session_state.score = None
    st.session_state.plan = None
    st.session_state.gemini_text = None
    st.session_state.answer_grade = None
    st.session_state.generated_test_cases = []


def _gemini_key_widget_key() -> str:
    version = int(st.session_state.get("gemini_api_key_widget_version", 0) or 0)
    return f"gemini_api_key_input_{version}"


def _queue_gemini_submit(flag_key: str) -> None:
    widget_key = _gemini_key_widget_key()
    st.session_state.pending_gemini_api_key = str(
        st.session_state.get(widget_key, "") or st.session_state.get("gemini_api_key", "") or ""
    )
    st.session_state.gemini_api_key = ""
    if widget_key in st.session_state:
        del st.session_state[widget_key]
    st.session_state.gemini_api_key_widget_version = int(
        st.session_state.get("gemini_api_key_widget_version", 0) or 0
    ) + 1
    st.session_state[flag_key] = True


@st.dialog("Unlock stronger optimization with Gemini")
def _gemini_key_dialog(source: str) -> None:
    st.write(
        "Gemini can search a wider optimization space and propose stronger candidate rewrites. "
        "Your code will still be locally validated and benchmarked before any generated code is accepted."
    )
    key = st.text_input(
        "Gemini API key",
        type="password",
        key=f"gemini_key_dialog_input_{source}",
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Use Gemini", key=f"use_gemini_{source}"):
            st.session_state.pending_gemini_api_key = key.strip()
            st.session_state.gemini_api_key = ""
            st.session_state.gemini_prompt_decision = "accepted"
            st.session_state.gemini_key_requested = False
            st.rerun()
    with col2:
        if st.button("Continue without Gemini", key=f"skip_gemini_{source}"):
            st.session_state.pending_gemini_api_key = ""
            st.session_state.gemini_prompt_decision = "declined"
            st.session_state.gemini_key_requested = False
            st.rerun()


def _resolve_gemini_key_for_action(source: str, fallback_key: str = "") -> Optional[str]:
    api_key = str(st.session_state.get("pending_gemini_api_key", "") or fallback_key or "").strip()
    if api_key:
        st.session_state.pending_gemini_api_key = ""
        st.session_state.gemini_prompt_decision = ""
        st.session_state.gemini_key_requested = False
        return api_key

    if st.session_state.get("gemini_prompt_decision") not in {"declined"}:
        st.session_state.gemini_key_requested = True
        _gemini_key_dialog(source)
        return None

    st.warning(
        "Continuing without Gemini. Complexity Lab will use its local optimizer, "
        "but Gemini can often generate broader rewrite candidates. Any candidate is still "
        "locally analyzed and benchmarked before it is shown."
    )
    return ""


def _validation_benchmark_input() -> str:
    if st.session_state.get("static_only_mode", False):
        return ""
    return str(st.session_state.get("benchmark_input", "") or "")


def _default_generated_benchmark_input() -> str:
    cases = st.session_state.get("generated_test_cases", [])
    if cases:
        return str(cases[0].benchmark_input or "")
    return ""


def _refresh_benchmark_input_for_entrypoint(force: bool = False) -> None:
    current_entrypoint = str(st.session_state.get("entrypoint", "") or "")
    last_entrypoint = str(st.session_state.get("benchmark_input_entrypoint", "") or "")

    generated_default = _default_generated_benchmark_input()
    if not generated_default:
        return

    current_input = str(st.session_state.get("benchmark_input", "") or "").strip()

    stale_entrypoint = current_entrypoint and current_entrypoint != last_entrypoint
    empty_input = not current_input
    looks_like_two_sum_demo = (
        '"args"' in current_input
        and "[2, 7, 11, 15" in current_input
        and "57" in current_input
    )

    if force or empty_input or stale_entrypoint or looks_like_two_sum_demo:
        st.session_state.benchmark_input = generated_default
        st.session_state.benchmark_input_entrypoint = current_entrypoint


def _sync_entrypoint_with_code(update_state: bool = True) -> List[str]:
    code = str(st.session_state.get("editor_code", "") or "")
    definitions = discover_entrypoints(code)
    if definitions and update_state:
        selected = choose_entrypoint(code, st.session_state.get("entrypoint", ""))
        if st.session_state.get("entrypoint") != selected:
            st.session_state.entrypoint = selected
    return [definition.callable_name for definition in definitions]


def _analyze_current(
    benchmark: Optional[BenchmarkResult] = None,
    candidate: Optional[OptimizedCodeCandidate] = None,
    prior_rejection_reasons: Optional[List[str]] = None,
) -> None:
    code = st.session_state.editor_code
    _sync_entrypoint_with_code(update_state=True)
    if len(code) > MAX_CODE_CHARS:
        st.error(f"Code is too large for this app demo. Limit: {MAX_CODE_CHARS:,} characters.")
        return
    analysis = analyze_code(code)
    score = calculate_optimization_score(analysis, benchmark)
    definitions = discover_entrypoints(code)
    st.session_state.generated_test_cases = generate_test_cases(
        code=code,
        entrypoint=st.session_state.entrypoint,
        definitions=definitions,
    )
    _refresh_benchmark_input_for_entrypoint()
    plan = build_optimization_plan(
        analysis,
        score,
        entrypoint=st.session_state.entrypoint,
        candidate=candidate,
        prior_rejection_reasons=prior_rejection_reasons,
        benchmark_input=_validation_benchmark_input(),
    )
    st.session_state.analysis = analysis
    st.session_state.score = score
    st.session_state.plan = plan


def _run_benchmark() -> Optional[BenchmarkResult]:
    if not should_run_auto_benchmark(st.session_state.static_only_mode):
        st.warning("Static-only mode is enabled, so benchmark execution is disabled.")
        return None
    settings = profile_settings(st.session_state.benchmark_profile)
    code = st.session_state.editor_code
    _sync_entrypoint_with_code(update_state=True)
    definitions = discover_entrypoints(code)
    st.session_state.generated_test_cases = generate_test_cases(
        code=code,
        entrypoint=st.session_state.entrypoint,
        definitions=definitions,
    )
    _refresh_benchmark_input_for_entrypoint()
    if st.session_state.docker_backend:
        benchmark = run_benchmark_in_docker(
            code=code,
            entrypoint=st.session_state.entrypoint,
            input_text=st.session_state.benchmark_input,
            repeat_count=st.session_state.repeat_count or settings["repeat_count"],
            warmup_count=settings["warmup_count"],
            timeout_seconds=st.session_state.timeout_seconds or settings["timeout_seconds"],
        )
    else:
        benchmark = run_benchmark(
            code=code,
            entrypoint=st.session_state.entrypoint,
            input_text=st.session_state.benchmark_input,
            repeat_count=st.session_state.repeat_count or settings["repeat_count"],
            warmup_count=settings["warmup_count"],
            timeout_seconds=st.session_state.timeout_seconds or settings["timeout_seconds"],
            allow_top_level=st.session_state.allow_top_level_benchmark,
        )
    st.session_state.benchmark = benchmark
    _analyze_current(benchmark)
    if benchmark.success:
        st.session_state.benchmark_history.append(
            {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "entrypoint": benchmark.entrypoint,
                "peak_ms": benchmark.summary.max_ms,
                "peak_kb": benchmark.summary.max_peak_memory_kb,
                "score": st.session_state.score.score if st.session_state.score else 0,
            }
        )
        st.session_state.benchmark_history = st.session_state.benchmark_history[-HISTORY_LIMIT:]
    return benchmark


def _run_scaling() -> None:
    if st.session_state.static_only_mode:
        st.warning("Static-only mode is enabled, so scaling benchmarks are disabled.")
        return
    settings = profile_settings(st.session_state.benchmark_profile)
    _sync_entrypoint_with_code(update_state=True)
    definitions = discover_entrypoints(st.session_state.editor_code)
    st.session_state.generated_test_cases = generate_test_cases(
        code=st.session_state.editor_code,
        entrypoint=st.session_state.entrypoint,
        definitions=definitions,
    )
    _refresh_benchmark_input_for_entrypoint()
    sizes = st.session_state.scaling_sizes
    if isinstance(sizes, str):
        parsed_sizes = [int(item.strip()) for item in sizes.split(",") if item.strip().isdigit()]
    else:
        parsed_sizes = list(settings["sizes"])
    scaling = run_scaling_benchmark(
        code=st.session_state.editor_code,
        entrypoint=st.session_state.entrypoint,
        input_text=st.session_state.benchmark_input,
        data_shape=st.session_state.scaling_shape,
        sizes=parsed_sizes or settings["sizes"],
        repeat_count=max(1, min(st.session_state.repeat_count, 10)),
        warmup_count=settings["warmup_count"],
        timeout_seconds=st.session_state.timeout_seconds,
    )
    st.session_state.scaling = scaling
    _analyze_current(st.session_state.benchmark)


def _generate_gemini_feedback(api_key: str) -> None:
    if not st.session_state.analysis or not st.session_state.score or not st.session_state.plan:
        _analyze_current(st.session_state.benchmark)
    if st.session_state.analysis and st.session_state.score and st.session_state.plan:
        st.session_state.gemini_text = enhance_with_gemini(
            api_key=api_key,
            code=st.session_state.editor_code,
            analysis=st.session_state.analysis,
            score=st.session_state.score,
            plan=st.session_state.plan,
        )


def _build_verified_optimization_plan(api_key: str) -> None:
    st.session_state.gemini_text = None
    benchmark = st.session_state.benchmark
    if st.session_state.static_only_mode:
        benchmark = None
        st.session_state.benchmark = None
        st.info("Static-only mode is enabled, so Generate Optimization Plan skipped automatic benchmarking.")
        _analyze_current(None)
    else:
        benchmark = _run_benchmark()

    analysis = st.session_state.analysis
    score = st.session_state.score
    seed_plan = st.session_state.plan
    if not analysis or not score or not seed_plan:
        return

    if not api_key:
        st.info(
            "No Gemini API key was provided. For broader code generation and deeper optimization, "
            "add a Gemini API key in the sidebar. For now, Complexity Lab will use its local "
            "deterministic optimizer and only show code that passes local validation."
        )
        _analyze_current(benchmark)
        return

    gemini_errors: List[str] = []
    rejection_reasons: List[str] = []
    accepted_candidate: Optional[OptimizedCodeCandidate] = None
    for retry_count in range(3):
        candidate, error = generate_optimized_code_with_gemini(
            api_key=api_key,
            code=st.session_state.editor_code,
            analysis=analysis,
            score=score,
            plan=seed_plan,
            entrypoint=st.session_state.entrypoint,
            retry_count=retry_count,
            rejection_reasons=rejection_reasons,
        )
        if error:
            gemini_errors.append(error)
            break
        _, validation = validate_optimized_candidate(
            analysis,
            score,
            candidate,
            entrypoint=st.session_state.entrypoint,
            benchmark_input=_validation_benchmark_input(),
        )
        if validation.status == "accepted":
            accepted_candidate = candidate
            break
        rejection_reasons = validation.rejection_reasons

    _analyze_current(
        benchmark,
        candidate=accepted_candidate,
        prior_rejection_reasons=gemini_errors + rejection_reasons,
    )
    plan = st.session_state.plan
    if plan and accepted_candidate and plan.validation.status == "accepted":
        st.session_state.gemini_text = plan.candidate_explanation or "Gemini generated the accepted optimized code candidate."
    elif gemini_errors or rejection_reasons:
        st.session_state.gemini_text = (
            "Gemini did not produce an accepted optimized rewrite. "
            "The app fell back to the validated local optimizer.\n\n"
            + "\n".join(f"- {item}" for item in (gemini_errors + rejection_reasons))
        )


def _save_current_record() -> None:
    analysis = st.session_state.analysis
    score = st.session_state.score
    plan = st.session_state.plan
    if not analysis or not score or not plan:
        st.warning("Run an analysis before saving to history.")
        return
    report = build_markdown_report(
        analysis,
        score,
        plan,
        st.session_state.benchmark,
        st.session_state.scaling,
        st.session_state.gemini_text,
    )
    record_id = save_analysis_record(
        session_name=st.session_state.practice_session,
        code=st.session_state.editor_code,
        entrypoint=st.session_state.entrypoint,
        benchmark_input=st.session_state.benchmark_input,
        analysis=analysis,
        score=score,
        plan=plan,
        benchmark=st.session_state.benchmark,
        scaling=st.session_state.scaling,
        report_markdown=report,
    )
    st.success(f"Saved analysis #{record_id} to {st.session_state.practice_session}.")


def _render_hero() -> None:
    st.markdown(
        f"""
<div class="hero">
  <h1>{APP_NAME}</h1>
  <p>{APP_TAGLINE}</p>
  <div class="hero-badges">
    <span class="pill">AST static analysis</span>
    <span class="pill">Empirical benchmarks</span>
    <span class="pill">Interview coaching</span>
    <span class="pill">Best-effort execution guardrails</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_summary_metrics(
    analysis: Optional[StaticAnalysisResult],
    score: Optional[ScoreBreakdown],
    benchmark: Optional[BenchmarkResult],
) -> None:
    if not analysis:
        return
    cols = st.columns(5)
    with cols[0]:
        st.markdown(
            metric_card("Time Complexity", analysis.estimated_time, "Estimated static AST heuristic"),
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            metric_card("Space Complexity", analysis.estimated_space, "Estimated auxiliary memory"),
            unsafe_allow_html=True,
        )
    with cols[2]:
        value = f"{score.score}/100" if score else "--"
        caption = f"{score.efficiency_percentage}% efficiency" if score else "Not scored"
        st.markdown(metric_card("Optimization Score", value, caption), unsafe_allow_html=True)
    with cols[3]:
        peak_ms = f"{benchmark.summary.max_ms:.3f} ms" if benchmark and benchmark.success else "Not run"
        st.markdown(metric_card("Peak Runtime", peak_ms, "Measured with perf_counter_ns"), unsafe_allow_html=True)
    with cols[4]:
        peak = f"{benchmark.summary.max_peak_memory_kb:.1f} KB" if benchmark and benchmark.success else "Not run"
        st.markdown(metric_card("Peak Memory", peak, "Measured with tracemalloc"), unsafe_allow_html=True)


def _render_static_tab(analysis: Optional[StaticAnalysisResult]) -> None:
    if not analysis:
        render_empty_state("Paste code and click Analyze Code to see complexity estimates.")
        return

    if not analysis.valid:
        st.error(analysis.parse_error or "The code could not be parsed.")
        for caveat in analysis.caveats:
            st.warning(caveat)
        return

    st.markdown(
        f"""
<div class="glass-card">
  <div class="section-title">Estimated Complexity</div>
  {complexity_badge("Time " + analysis.estimated_time)}
  {complexity_badge("Space " + analysis.estimated_space)}
  <span class="pill">Confidence {int(analysis.confidence * 100)}%</span>
  <p class="muted" style="margin-top:12px;">
    These estimates are static heuristics based on AST evidence, not mathematical proof.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    if analysis.confidence_breakdown:
        confidence_cols = st.columns(len(analysis.confidence_breakdown))
        for column, (name, value) in zip(confidence_cols, analysis.confidence_breakdown.items()):
            with column:
                st.metric(name.replace("_", " ").title(), f"{int(value * 100)}%")

    if analysis.algorithm_patterns:
        st.subheader("Algorithm Pattern Recognition")
        for algorithm_pattern in analysis.algorithm_patterns[:5]:
            st.markdown(
                f"""
<div class="glass-card">
  <div class="section-title">{html.escape(algorithm_pattern.name)} <span class="pill">{int(algorithm_pattern.confidence * 100)}% match</span></div>
  <div class="muted">{html.escape(algorithm_pattern.interview_note)}</div>
  <p><strong>Signals:</strong> {html.escape(", ".join(algorithm_pattern.evidence))}</p>
</div>
""",
                unsafe_allow_html=True,
            )

    function_rows = [
        {
            "Scope": item.name,
            "Line": item.lineno,
            "Time Complexity": item.estimated_time,
            "Space Complexity": item.estimated_space,
            "Confidence": f"{int(item.confidence * 100)}%",
        }
        for item in analysis.functions
    ]
    st.dataframe(function_rows, width="stretch", hide_index=True)

    if analysis.line_findings:
        st.subheader("Line-Level Findings")
        st.dataframe(
            [
                {
                    "Line": item.line,
                    "Severity": item.severity,
                    "Title": item.title,
                    "Category": item.category,
                    "Suggestion": item.suggestion,
                }
                for item in analysis.line_findings
            ],
            width="stretch",
            hide_index=True,
        )

    generated_cases = st.session_state.get("generated_test_cases", [])
    if generated_cases:
        st.subheader("Generated Test Cases")
        st.caption("These are locally generated from the detected entrypoint and code shape.")

        for index, case in enumerate(generated_cases, start=1):
            with st.expander(f"Test case {index}: {case.name}", expanded=index == 1):
                st.code(case.benchmark_input, language="json")
                if case.expected_output:
                    st.write(f"Expected output: `{case.expected_output}`")
                if case.reason:
                    st.caption(case.reason)

                if st.button(
                    f"Use as benchmark input #{index}",
                    key=f"use_generated_case_{index}",
                ):
                    st.session_state.benchmark_input = case.benchmark_input
                    st.session_state.benchmark_input_entrypoint = st.session_state.entrypoint
                    st.success("Benchmark input updated from generated test case.")
                    _rerun()

    if analysis.anti_patterns:
        st.subheader("Detected Bottlenecks")
        for anti_pattern in analysis.anti_patterns:
            st.markdown(
                f"""
<div class="glass-card">
  <div class="section-title">{html.escape(anti_pattern.name)} {severity_pill(anti_pattern.severity)}</div>
  <div class="muted">{html.escape(anti_pattern.message)}</div>
  <p><strong>Evidence:</strong> {html.escape(anti_pattern.evidence)}</p>
  <p><strong>Suggested target:</strong> {html.escape(anti_pattern.suggestion)}</p>
</div>
""",
                unsafe_allow_html=True,
            )
    else:
        st.success("No major static-analysis bottleneck was detected.")

    with st.expander("Evidence"):
        for item in analysis.evidence:
            st.write(f"- {item}")
    with st.expander("Caveats"):
        for item in analysis.caveats:
            st.write(f"- {item}")
    with st.expander("Raw static metrics"):
        st.json(analysis.metrics)
    with st.expander("Function call graph"):
        st.json(analysis.call_graph)


def _render_benchmark_tab(
    benchmark: Optional[BenchmarkResult],
    scaling: Optional[ScalingBenchmarkResult],
) -> None:
    if not benchmark:
        render_empty_state("Run Benchmarks to collect empirical timing and memory data.")
    elif not benchmark.success:
        st.error(benchmark.error or "Benchmark failed.")
        st.info("Static analysis is still available even when execution is blocked or fails.")
    else:
        cols = st.columns(4)
        cols[0].metric("Min runtime", f"{benchmark.summary.min_ms:.4f} ms")
        cols[1].metric("Peak Runtime", f"{benchmark.summary.max_ms:.4f} ms")
        cols[2].metric("Std dev", f"{benchmark.summary.std_ms:.4f} ms")
        cols[3].metric("Peak memory", f"{benchmark.summary.max_peak_memory_kb:.2f} KB")
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.plotly_chart(runtime_chart(benchmark))
        with chart_cols[1]:
            st.plotly_chart(memory_chart(benchmark))
        st.dataframe([run.__dict__ for run in benchmark.runs], width="stretch", hide_index=True)

    st.subheader("Benchmark History")
    history = st.session_state.benchmark_history
    if history:
        st.plotly_chart(history_chart(history))
        st.dataframe(history, width="stretch", hide_index=True)
    else:
        st.caption("No successful benchmark runs in this session yet.")

    st.subheader("Input-Size Scaling Experiment")
    if scaling:
        if scaling.success:
            st.markdown(
                f"""
<div class="glass-card">
  <div class="section-title">Empirical Curve Fit {complexity_badge(scaling.empirical_complexity)}</div>
  <p class="muted">This is a curve fit over generated benchmark inputs, not a proof of asymptotic complexity.</p>
</div>
""",
                unsafe_allow_html=True,
            )
            st.plotly_chart(scaling_chart(scaling))
            st.dataframe([point.__dict__ for point in scaling.points], width="stretch", hide_index=True)
            with st.expander("Curve fit R2 scores"):
                st.json(scaling.fit_scores)
        else:
            st.error(scaling.error or "Scaling benchmark failed.")
    else:
        st.caption("Run Scaling Benchmark to compare runtime and memory across generated input sizes.")

    with st.expander("Execution safety notes"):
        notes = benchmark.safety_notes if benchmark else [
            "Benchmarks use a best-effort restricted execution context, not a secure sandbox.",
            "Dangerous imports and builtins are blocked before execution when detected.",
            "Timeouts are enforced with a separate worker process.",
        ]
        for note in notes:
            st.write(f"- {note}")


def _render_step(step: OptimizationStep) -> None:
    st.markdown(
        f"""
<div class="glass-card">
  <div class="section-title">{html.escape(step.title)} {severity_pill(step.priority)}</div>
  <p><strong>Why:</strong> {html.escape(step.why)}</p>
  <p><strong>Change:</strong> {html.escape(step.change)}</p>
  <p><strong>Runtime effect:</strong> {html.escape(step.runtime_effect)}</p>
  <p><strong>Memory effect:</strong> {html.escape(step.memory_effect)}</p>
  <p><strong>Expected complexity change:</strong> {html.escape(step.expected_complexity_change or "Clarifies performance reasoning")}</p>
  <p><strong>Line evidence:</strong> {html.escape(str(step.line) if step.line else "Pattern-level finding")}</p>
  <p><strong>Interview note:</strong> {html.escape(step.interview_note)}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_optimization_tab(
    analysis: Optional[StaticAnalysisResult],
    score: Optional[ScoreBreakdown],
    plan: Optional[OptimizationPlan],
) -> None:
    if not analysis or not score or not plan:
        render_empty_state("Generate an analysis to see the optimization plan.")
        return

    st.markdown(
        f"""
<div class="glass-card">
  <div class="section-title">Optimization Assessment {severity_pill(score.severity)}</div>
  <p class="muted">{html.escape(plan.summary)}</p>
  <p><strong>Improvement potential:</strong> {html.escape(score.improvement_potential)}</p>
</div>
""",
        unsafe_allow_html=True,
    )
    validation = plan.validation
    if validation.status == "accepted" and validation.candidate_score < validation.original_score:
        st.error(
            "Internal validation error: candidate was accepted even though its score decreased. "
            "This candidate will not be displayed."
        )
        plan.optimized_code = None
        validation.status = "rejected"
        validation.accepted_reason = ""
        validation.rejection_reasons.append(
            "Internal validation guard blocked an accepted candidate with a lower optimization score."
        )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Current Time", analysis.estimated_time)
    metric_cols[1].metric("Current Space", analysis.estimated_space)
    metric_cols[2].metric("Current Score", f"{score.score}/100")
    metric_cols[3].metric("Candidate Time", validation.candidate_time or "Not accepted")
    metric_cols[4].metric("Candidate Space", validation.candidate_space or "Not accepted")

    already_optimal_linear = (
        score.score >= 95
        and analysis.estimated_time == "O(n)"
        and analysis.estimated_space == "O(1)"
    )
    if already_optimal_linear and validation.status != "accepted":
        st.info(
            "The current solution is already asymptotically optimal. "
            "No lower time/space complexity is available for the general problem. "
            "The app will show a cleaner reference implementation when one passes validation."
        )

    status_label = validation.status.replace("_", " ").title()
    status_detail = validation.accepted_reason or "No verified optimized code is available yet."
    if already_optimal_linear and validation.status != "accepted":
        status_detail = (
            "Current solution is already asymptotically optimal. Showing a clean reference implementation "
            "requires a same-complexity candidate that passes local validation."
        )
    st.markdown(
        f"""
<div class="glass-card">
  <div class="section-title">Verified Code Generation {severity_pill(status_label)}</div>
  <p><strong>Source:</strong> {html.escape(validation.source.title())}</p>
  <p><strong>Status:</strong> {html.escape(status_label)}</p>
  <p class="muted">{html.escape(status_detail)}</p>
  <p><strong>Before:</strong> {html.escape(validation.original_time or analysis.estimated_time)} time, {html.escape(validation.original_space or analysis.estimated_space)} space, {validation.original_score or score.score}/100 score.</p>
  <p><strong>After:</strong> {html.escape(validation.candidate_time or "Not accepted")} time, {html.escape(validation.candidate_space or "Not accepted")} space, {validation.candidate_score or 0}/100 score.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    if validation.memory_tradeoff:
        st.warning(
            f"The verified rewrite improves time but increases estimated space from {validation.original_space} to {validation.candidate_space}."
        )
    if validation.rejection_reasons:
        with st.expander("Rejected candidate reasons and fallback notes"):
            for reason in validation.rejection_reasons:
                st.write(f"- {reason}")

    st.subheader("Verified Optimization Candidates")
    if plan.tiered_candidates:
        tier_columns = st.columns(3)
        for column, candidate in zip(tier_columns, plan.tiered_candidates):
            with column:
                st.markdown(f"##### {candidate.tier}")
                expander_title = candidate.title.replace(f"{candidate.tier}: ", "")
                with st.expander(expander_title, expanded=candidate.accepted):
                    st.write(candidate.explanation)
                    st.write(f"Expected time: `{candidate.expected_time}`")
                    st.write(f"Expected space: `{candidate.expected_space}`")
                    if candidate.benchmark_comparison:
                        comparison = candidate.benchmark_comparison
                        st.write(
                            f"Original avg: {comparison.original_avg_ms:.4f} ms; "
                            f"Candidate avg: {comparison.candidate_avg_ms:.4f} ms"
                        )
                        st.write(
                            f"Original peak memory: {comparison.original_peak_memory_kb:.2f} KB; "
                            f"Candidate peak memory: {comparison.candidate_peak_memory_kb:.2f} KB"
                        )
                    if candidate.accepted:
                        st.success("Accepted")
                        st.code(candidate.code, language="python")
                    else:
                        st.warning(candidate.rejection_reason or "Rejected because it did not beat the original.")
    else:
        st.info("No verified optimization candidates were generated for the current code.")

    if plan.step_by_step_plan:
        st.subheader("Step-by-Step Optimization Plan")
        for index, item in enumerate(plan.step_by_step_plan, start=1):
            st.write(f"{index}. {item}")

    chart_cols = st.columns([1, 1])
    with chart_cols[0]:
        st.plotly_chart(score_gauge(score))
    with chart_cols[1]:
        st.plotly_chart(score_breakdown_chart(score))

    step_tabs = st.tabs(["Quick Wins", "Medium Refactors", "Advanced Improvements"])
    with step_tabs[0]:
        for step in plan.quick_wins:
            _render_step(step)
    with step_tabs[1]:
        if plan.medium_refactors:
            for step in plan.medium_refactors:
                _render_step(step)
        else:
            st.caption("No medium refactor was detected from current static signals.")
    with step_tabs[2]:
        if plan.advanced_improvements:
            for step in plan.advanced_improvements:
                _render_step(step)
        else:
            st.caption("No advanced rewrite was detected from current static signals.")

    st.subheader("Original vs Best Verified Suggestion")
    before, after = st.columns(2)
    with before:
        st.caption("Original")
        st.code(analysis.raw_code, language="python")
    with after:
        st.caption("Suggested")
        if plan.optimized_code and plan.validation.status == "accepted":
            st.caption(f"Source: {plan.validation.source.title()}")
            st.caption(f"Safe rewrite confidence: {int(plan.safe_rewrite_confidence * 100)}%")
            st.code(plan.optimized_code, language="python")
        else:
            st.info(
                "No verified better rewrite was found. The original code remains the best validated version "
                "for the current benchmark input."
            )
    st.markdown(f"**Before/After:** {plan.before_after}")
    if plan.rewrite_tests:
        with st.expander("Suggested validation tests for rewrite"):
            st.code("\n".join(plan.rewrite_tests), language="python")
    with st.expander("Manual optimization checklist"):
        for item in plan.manual_checklist:
            st.write(f"- {item}")
    with st.expander("Trade-off notes"):
        for note in plan.tradeoffs:
            st.write(f"- {note}")


def _render_interview_tab(
    analysis: Optional[StaticAnalysisResult],
    plan: Optional[OptimizationPlan],
    gemini_text: Optional[str],
    answer_grade: Optional[InterviewGrade],
) -> None:
    if not analysis or not plan:
        render_empty_state("Generate an analysis to unlock interview mode.")
        return
    feedback = plan.interview_feedback
    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            metric_card("Interview Readiness", feedback.get("acceptability", ""), feedback.get("likely_concern", "")),
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            metric_card("Primary Message", "Explain trade-offs", "Asymptotic estimate + empirical benchmark caveat"),
            unsafe_allow_html=True,
        )

    st.markdown("#### How to explain your current solution")
    st.info(feedback.get("current_explanation", ""))
    st.markdown("#### How to explain the optimized approach")
    st.success(feedback.get("optimized_explanation", ""))
    st.markdown("#### Trade-offs to mention")
    st.warning(feedback.get("tradeoffs", ""))

    spoken = (
        f"{feedback.get('current_explanation', '')} "
        f"{feedback.get('optimized_explanation', '')} "
        "I would validate the change with benchmarks, but I would still use asymptotic complexity as the main interview argument."
    )
    copyable_block("Copyable Interview Answer", spoken, "interview")

    st.markdown("#### Timed Interview Simulation")
    timer_cols = st.columns(3)
    with timer_cols[0]:
        st.metric("Suggested answer window", "90 sec")
    with timer_cols[1]:
        st.metric("Follow-up questions", str(len(build_follow_up_questions(analysis, plan))))
    with timer_cols[2]:
        st.metric("Rubric categories", "5")

    st.markdown("#### Interviewer follow-up questions")
    for question in build_follow_up_questions(analysis, plan):
        st.write(f"- {question}")

    st.text_area(
        "Write your interview answer and grade it locally",
        key="interview_answer",
        height=160,
        help="This local rubric does not call Gemini. It checks correctness language, complexity, edge cases, trade-offs, and communication length.",
    )
    if st.button("Grade My Answer", width="stretch"):
        st.session_state.answer_grade = grade_interview_answer(st.session_state.interview_answer, analysis, plan)
        answer_grade = st.session_state.answer_grade
    if answer_grade:
        st.markdown("#### Local Answer Grade")
        st.metric("Total score", f"{answer_grade.total_score}/100")
        st.dataframe(
            [{"Rubric": key, "Points": value} for key, value in answer_grade.rubric.items()],
            width="stretch",
            hide_index=True,
        )
        cols = st.columns(2)
        with cols[0]:
            st.success("\n".join(f"- {item}" for item in answer_grade.strengths) or "No strengths detected yet.")
        with cols[1]:
            st.warning("\n".join(f"- {item}" for item in answer_grade.improvements) or "No major gaps detected.")
        with st.expander("Model answer"):
            st.write(answer_grade.model_answer)

    if gemini_text:
        st.markdown("#### Gemini-enhanced coaching")
        st.markdown(gemini_text)


def _render_report_tab(
    analysis: Optional[StaticAnalysisResult],
    score: Optional[ScoreBreakdown],
    plan: Optional[OptimizationPlan],
    benchmark: Optional[BenchmarkResult],
    scaling: Optional[ScalingBenchmarkResult],
    gemini_text: Optional[str],
) -> None:
    report = build_markdown_report(analysis, score, plan, benchmark, scaling, gemini_text)
    html_report = build_html_report(analysis, score, plan, benchmark, scaling, gemini_text)
    download_cols = st.columns(2)
    with download_cols[0]:
        st.download_button(
            "Download Markdown Report",
            data=report,
            file_name="complexity_lab_report.md",
            mime="text/markdown",
            width="stretch",
        )
    with download_cols[1]:
        st.download_button(
            "Download HTML Report",
            data=html_report,
            file_name="complexity_lab_report.html",
            mime="text/html",
            width="stretch",
        )
    if analysis and score:
        copyable_block("Copy for LinkedIn / GitHub README", build_linkedin_summary(analysis, score, benchmark), "linkedin")
    copyable_block("Copyable Executive Summary", report[:3500], "report")
    with st.expander("Full markdown report preview"):
        st.markdown(report)


def _render_algorithm_planner_result(result: AlgorithmPlannerResult) -> None:
    if result.error:
        if result.valid:
            st.warning(result.error)
        else:
            st.error(result.error)
    if result.safety_error:
        st.info(result.safety_error)

    st.subheader("Problem Understanding")
    st.write(result.problem_understanding or "Not available.")

    st.subheader("Step-by-Step Optimization Plan")
    if result.step_by_step_optimization_plan:
        for index, step in enumerate(result.step_by_step_optimization_plan, start=1):
            st.write(f"{index}. {step}")
    else:
        st.caption("No step-by-step plan was generated.")

    st.subheader("Best Data Structure / Algorithm Choice")
    st.info(result.best_data_structure_algorithm_choice or "Not available.")

    st.subheader("Final Optimized Python Code")
    if result.final_optimized_python_code:
        st.code(result.final_optimized_python_code, language="python")
    else:
        st.info("Not generated in local mode or blocked by safety validation.")

    complexity_cols = st.columns(3)
    with complexity_cols[0]:
        st.markdown(
            metric_card("Time Complexity", result.time_complexity or "Not available", "Big-O analysis"),
            unsafe_allow_html=True,
        )
    with complexity_cols[1]:
        st.markdown(
            metric_card("Space Complexity", result.space_complexity or "Not available", "Big-O analysis"),
            unsafe_allow_html=True,
        )
    with complexity_cols[2]:
        st.markdown(
            metric_card("Peak Runtime", result.runtime.display_value, "Measured only when safe test cases run"),
            unsafe_allow_html=True,
        )
    st.caption(result.complexity_note)
    if result.runtime.details:
        with st.expander("Peak runtime measurement details"):
            for detail in result.runtime.details:
                st.write(f"- {detail}")

    st.subheader("Test Cases")
    if result.test_cases:
        st.dataframe(
            [
                {
                    "Name": test_case.name,
                    "Input": test_case.input_text,
                    "Expected Output": test_case.expected_output,
                }
                for test_case in result.test_cases
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No executable test cases were generated.")


def _render_algorithm_planner_tab() -> None:
    st.markdown("#### Algorithm Optimization Planner")
    question = st.text_area(
        "Enter your coding question",
        key="algorithm_planner_question",
        height=220,
        placeholder="Paste a coding interview problem, constraints, examples, and expected output.",
    )
    submitted = st.button(
        "Generate Optimization Plan",
        type="primary",
        key="algorithm_planner_generate_plan",
        width="stretch",
        on_click=_queue_gemini_submit,
        args=("algorithm_planner_submit_pending",),
    )

    pending_submit = bool(st.session_state.get("algorithm_planner_submit_pending", False))
    if submitted or pending_submit:
        current_api_key = _resolve_gemini_key_for_action(
            "algorithm_planner",
            str(st.session_state.get("gemini_api_key", "") or ""),
        )
        if current_api_key is None:
            return
        st.session_state.algorithm_planner_submit_pending = False
        current_question = str(st.session_state.get("algorithm_planner_question", question) or "")
        with st.spinner("Generating optimization plan..."):
            st.session_state.algorithm_planner_result = generate_algorithm_optimization_plan(
                current_question,
                current_api_key,
            )

    result = st.session_state.algorithm_planner_result
    if result:
        if result.source == "local":
            st.info("No Gemini API key was provided, so this is a local estimated planning outline.")
        _render_algorithm_planner_result(result)
    else:
        render_empty_state("Enter a coding question and generate an optimization plan.")


def _render_code_analyzer_workflow(gemini_api_key: str) -> None:
    st.markdown("#### Python Code")
    st.text_area(
        "Paste Python code",
        key="editor_code",
        height=360,
        label_visibility="collapsed",
    )

    action_cols = st.columns([1, 1, 1, 1, 1])
    with action_cols[0]:
        analyze_clicked = st.button("Analyze Code", type="primary", width="stretch")
    with action_cols[1]:
        benchmark_clicked = st.button("Run Benchmarks", width="stretch")
    with action_cols[2]:
        scaling_clicked = st.button("Run Scaling Benchmark", width="stretch")
    with action_cols[3]:
        plan_clicked = st.button(
            "Generate Optimization Plan",
            key="code_analyzer_generate_plan",
            width="stretch",
            on_click=_queue_gemini_submit,
            args=("code_analyzer_plan_submit_pending",),
        )
    with action_cols[4]:
        save_clicked = st.button("Save Analysis", width="stretch")

    if analyze_clicked:
        with st.spinner("Analyzing AST features and estimating complexity..."):
            st.session_state.benchmark = None
            st.session_state.gemini_text = None
            _analyze_current(None)
    if benchmark_clicked:
        with st.spinner("Running guarded benchmarks with timing and memory tracing..."):
            st.session_state.gemini_text = None
            _run_benchmark()
    if scaling_clicked:
        with st.spinner("Running generated input-size scaling experiments..."):
            st.session_state.gemini_text = None
            _run_scaling()
    pending_plan_click = bool(st.session_state.get("code_analyzer_plan_submit_pending", False))
    if plan_clicked or pending_plan_click:
        current_api_key = _resolve_gemini_key_for_action("code_analyzer", gemini_api_key)
        if current_api_key is None:
            return
        st.session_state.code_analyzer_plan_submit_pending = False
        with st.spinner("Analyzing code, collecting benchmark metrics, and validating optimized candidates..."):
            _build_verified_optimization_plan(current_api_key)
    if save_clicked:
        _save_current_record()

    analysis_state = st.session_state.analysis
    benchmark_state = st.session_state.benchmark
    scaling_state = st.session_state.scaling
    score_state = st.session_state.score
    plan_state = st.session_state.plan
    gemini_state = st.session_state.gemini_text
    answer_grade_state = st.session_state.answer_grade

    _render_summary_metrics(analysis_state, score_state, benchmark_state)

    tabs = st.tabs(["Static Analysis", "Benchmarks", "Optimization Plan", "Interview Mode", "Report", "Progress"])
    with tabs[0]:
        _render_static_tab(analysis_state)
    with tabs[1]:
        _render_benchmark_tab(benchmark_state, scaling_state)
    with tabs[2]:
        _render_optimization_tab(analysis_state, score_state, plan_state)
    with tabs[3]:
        _render_interview_tab(analysis_state, plan_state, gemini_state, answer_grade_state)
    with tabs[4]:
        _render_report_tab(analysis_state, score_state, plan_state, benchmark_state, scaling_state, gemini_state)
    with tabs[5]:
        _render_progress_tab()


def _render_progress_tab() -> None:
    session_name = st.session_state.practice_session
    summary = progress_summary(session_name)
    cols = st.columns(4)
    cols[0].metric("Saved analyses", summary["count"])
    cols[1].metric("Average score", summary["average_score"])
    cols[2].metric("Best score", summary["best_score"])
    cols[3].metric("Latest score", summary["latest_score"])
    st.caption(f"Weakest recurring severity: {summary['weakest_area']}")
    records = load_recent_records(limit=30, session_name=session_name)
    if records:
        st.dataframe(records, width="stretch", hide_index=True)
    else:
        st.info("No saved analyses for this practice session yet. Run an analysis and click Save Analysis.")


def main() -> None:
    _initialize_state()
    inject_global_styles()

    with st.sidebar:
        st.title(APP_NAME)
        st.caption("Controls")
        st.selectbox(
            "Practice session",
            ["Arrays", "Strings", "Hashmaps", "Dynamic Programming", "Graphs", "Pandas/Numpy", "General"],
            key="practice_session",
        )
        example_name = st.selectbox("Example template", list(EXAMPLES.keys()))
        load_col, reset_col = st.columns(2)
        with load_col:
            if st.button("Load Example", width="stretch"):
                example = get_example(example_name)
                st.session_state.editor_code = example["code"]
                st.session_state.benchmark_input = example["input"]
                st.session_state.entrypoint = example["entrypoint"]
                st.session_state.benchmark_input_entrypoint = example["entrypoint"]
                _clear_outputs()
                _rerun()
        with reset_col:
            if st.button("Reset", width="stretch"):
                st.session_state.editor_code = DEFAULT_CODE
                st.session_state.benchmark_input = DEFAULT_INPUT
                st.session_state.entrypoint = "two_sum"
                st.session_state.benchmark_input_entrypoint = "two_sum"
                st.session_state.benchmark_history = []
                _clear_outputs()
                _rerun()

        st.divider()
        st.checkbox(
            "Static-only public mode",
            key="static_only_mode",
            help="Disable code execution for public demos or untrusted snippets.",
        )
        st.checkbox(
            "Allow top-level script benchmarking",
            key="allow_top_level_benchmark",
            value=False,
            help="Off by default. Prefer benchmarking a named function entrypoint.",
        )
        st.checkbox(
            "Use Docker isolation if available",
            key="docker_backend",
            help="Optional local/pro deployment mode. Streamlit Community Cloud usually does not provide Docker.",
        )
        if st.session_state.docker_backend:
            st.caption("Docker available: " + ("yes" if docker_available() else "no"))
        st.selectbox("Benchmark profile", ["Quick", "Balanced", "Stress"], index=1, key="benchmark_profile")
        entrypoint_options = _sync_entrypoint_with_code()
        if entrypoint_options:
            current_entrypoint = st.session_state.entrypoint
            st.selectbox(
                "Entrypoint function",
                entrypoint_options,
                index=entrypoint_options.index(current_entrypoint) if current_entrypoint in entrypoint_options else 0,
                key="entrypoint",
                help="Auto-detected from the pasted code. Class methods are shown as ClassName.method.",
            )
            st.caption("Auto-detected callable entrypoints: " + ", ".join(entrypoint_options))
        else:
            st.text_input(
                "Entrypoint function",
                key="entrypoint",
                help="No functions were detected. Leave blank to benchmark top-level script execution.",
            )
        definitions = discover_entrypoints(str(st.session_state.get("editor_code", "") or ""))
        st.session_state.generated_test_cases = generate_test_cases(
            code=str(st.session_state.get("editor_code", "") or ""),
            entrypoint=st.session_state.entrypoint,
            definitions=definitions,
        )
        _refresh_benchmark_input_for_entrypoint()
        st.text_area(
            "Benchmark input",
            key="benchmark_input",
            height=140,
            help='Use JSON/Python literal, {"args": [...], "kwargs": {...}}, or simple assignments like arr = [1, 2, 3].',
        )
        st.slider("Benchmark repeats", min_value=1, max_value=30, value=DEFAULT_REPEAT_COUNT, key="repeat_count")
        st.slider(
            "Timeout seconds",
            min_value=1.0,
            max_value=20.0,
            value=DEFAULT_TIMEOUT_SECONDS,
            step=0.5,
            key="timeout_seconds",
        )
        st.selectbox("Scaling input shape", ["list", "string", "matrix", "dict", "graph"], key="scaling_shape")
        st.text_input("Scaling sizes", value="10,100,500", key="scaling_sizes")
        st.select_slider("Input-size mindset", options=["Tiny", "Small", "Medium", "Large", "Stress"], value="Small")
        gemini_api_key = st.text_input(
            "Gemini API key (optional)",
            key=_gemini_key_widget_key(),
            type="password",
            help=(
                "Used only for the current Streamlit request to enhance feedback and the Algorithm Planner. "
                "It is not saved to history or reports."
            ),
        )
        st.session_state.gemini_api_key = gemini_api_key
        st.caption("Execution protections are best-effort and intended for interview-prep snippets.")

    _render_hero()

    workflow_tabs = st.tabs(["Algorithm Planner", "Code Analyzer"])
    with workflow_tabs[0]:
        _render_algorithm_planner_tab()
    with workflow_tabs[1]:
        _render_code_analyzer_workflow(gemini_api_key)


if __name__ == "__main__":
    main()
