"""Markdown report generation."""

from __future__ import annotations

import html
from typing import Optional

from analyzer.models import StaticAnalysisResult
from benchmarking.metrics import BenchmarkResult, ScalingBenchmarkResult
from optimization.planner import OptimizationPlan
from scoring.optimizer_score import ScoreBreakdown


def _bullet(items: list) -> str:
    if not items:
        return "- None"
    return "\n".join(f"- {item}" for item in items)


def build_markdown_report(
    analysis: Optional[StaticAnalysisResult],
    score: Optional[ScoreBreakdown],
    plan: Optional[OptimizationPlan],
    benchmark: Optional[BenchmarkResult],
    scaling: Optional[ScalingBenchmarkResult] = None,
    gemini_text: Optional[str] = None,
) -> str:
    if not analysis:
        return "# Complexity Lab Report\n\nNo analysis has been generated yet.\n"

    benchmark_section = "Benchmark was not run."
    if benchmark:
        if benchmark.success:
            benchmark_section = (
                f"- Entrypoint: `{benchmark.entrypoint}`\n"
                f"- Repeats: {benchmark.summary.repeat_count}\n"
                f"- Peak runtime: {benchmark.summary.max_ms:.4f} ms\n"
                f"- Min/Peak runtime: {benchmark.summary.min_ms:.4f} / {benchmark.summary.max_ms:.4f} ms\n"
                f"- Max peak traced memory: {benchmark.summary.max_peak_memory_kb:.2f} KB"
            )
        else:
            benchmark_section = f"Benchmark did not run successfully: {benchmark.error}"

    score_section = "Score was not calculated."
    if score:
        components = "\n".join(f"- {name}: {value}" for name, value in score.components.items())
        score_section = (
            f"- Score: {score.score}/100\n"
            f"- Efficiency: {score.efficiency_percentage}%\n"
            f"- Severity: {score.severity}\n"
            f"- Improvement potential: {score.improvement_potential}\n\n"
            f"### Score Components\n{components}"
        )

    plan_section = "Optimization plan was not generated."
    if plan:
        quick = _bullet([step.title + ": " + step.change for step in plan.quick_wins])
        medium = _bullet([step.title + ": " + step.change for step in plan.medium_refactors])
        advanced = _bullet([step.title + ": " + step.change for step in plan.advanced_improvements])
        validation = plan.validation
        rejected = _bullet(validation.rejection_reasons)
        step_plan = _bullet(plan.step_by_step_plan)
        plan_section = (
            f"{plan.summary}\n\n"
            f"### Quick Wins\n{quick}\n\n"
            f"### Medium Refactors\n{medium}\n\n"
            f"### Advanced Improvements\n{advanced}\n\n"
            f"### Verified Code Generation\n"
            f"- Source: {validation.source}\n"
            f"- Status: {validation.status}\n"
            f"- Before: {validation.original_time or analysis.estimated_time} time, {validation.original_space or analysis.estimated_space} space, {validation.original_score or (score.score if score else 0)}/100 score\n"
            f"- After: {validation.candidate_time or 'Not accepted'} time, {validation.candidate_space or 'Not accepted'} space, {validation.candidate_score or 0}/100 score\n"
            f"- Memory trade-off: {'yes' if validation.memory_tradeoff else 'no'}\n\n"
            f"### Step-by-Step Generated Plan\n{step_plan}\n\n"
            f"### Rejected Candidate Reasons\n{rejected}\n\n"
            f"### Interview Feedback\n"
            f"- Acceptability: {plan.interview_feedback.get('acceptability', '')}\n"
            f"- Concern: {plan.interview_feedback.get('likely_concern', '')}\n"
            f"- Current explanation: {plan.interview_feedback.get('current_explanation', '')}"
        )

    gemini_section = gemini_text or "Gemini enhancement was not requested."
    patterns = _bullet(
        [
            f"{pattern.name} ({int(pattern.confidence * 100)}%): {pattern.interview_note}"
            for pattern in analysis.algorithm_patterns
        ]
    )
    line_findings = _bullet(
        [
            f"Line {finding.line}: {finding.title} [{finding.severity}] - {finding.suggestion}"
            for finding in analysis.line_findings
        ]
    )
    confidence = "\n".join(
        f"- {name.replace('_', ' ').title()}: {int(value * 100)}%"
        for name, value in analysis.confidence_breakdown.items()
    ) or "- Not available"
    scaling_section = "Scaling benchmark was not run."
    if scaling:
        if scaling.success:
            points = "\n".join(
                f"- n={point.input_size}: peak {point.max_ms:.4f} ms, {point.max_peak_memory_kb:.2f} KB"
                for point in scaling.points
                if point.success
            )
            fits = "\n".join(f"- {name}: R2 {value:.3f}" for name, value in scaling.fit_scores.items())
            scaling_section = (
                f"- Empirical fit: `{scaling.empirical_complexity}`\n"
                f"- Data shape: `{scaling.data_shape}`\n\n"
                f"### Scaling Points\n{points or '- None'}\n\n"
                f"### Curve Fit Scores\n{fits or '- Not enough data'}"
            )
        else:
            scaling_section = f"Scaling benchmark failed: {scaling.error}"

    return f"""# Complexity Lab Report

## Static Analysis
- Time complexity: `{analysis.estimated_time}` (estimated from static analysis)
- Space complexity: `{analysis.estimated_space}` (estimated from static analysis)
- Confidence: {int(analysis.confidence * 100)}%

### Confidence Breakdown
{confidence}

### Algorithm Pattern Recognition
{patterns}

### Line-Level Findings
{line_findings}

### Evidence
{_bullet(analysis.evidence)}

### Caveats
{_bullet(analysis.caveats)}

## Benchmark Results
{benchmark_section}

## Scaling Benchmark
{scaling_section}

## Optimization Score
{score_section}

## Optimization Plan
{plan_section}

## Gemini-Enhanced Coaching
{gemini_section}

## Source Code
```python
{analysis.raw_code}
```
"""


def build_linkedin_summary(
    analysis: Optional[StaticAnalysisResult],
    score: Optional[ScoreBreakdown],
    benchmark: Optional[BenchmarkResult],
) -> str:
    if not analysis or not score:
        return "Complexity Lab analyzes Python interview solutions with static complexity estimates and empirical benchmarks."
    runtime = "benchmark not run"
    if benchmark and benchmark.success:
        runtime = f"peak runtime {benchmark.summary.max_ms:.3f} ms, peak traced memory {benchmark.summary.max_peak_memory_kb:.1f} KB"
    return (
        f"Analyzed a Python solution with Complexity Lab: estimated {analysis.estimated_time} time, "
        f"{analysis.estimated_space} space, optimization score {score.score}/100, {runtime}. "
        "The report separates AST-based estimates from measured benchmark data and includes interview-ready optimization guidance."
    )


def build_html_report(
    analysis: Optional[StaticAnalysisResult],
    score: Optional[ScoreBreakdown],
    plan: Optional[OptimizationPlan],
    benchmark: Optional[BenchmarkResult],
    scaling: Optional[ScalingBenchmarkResult] = None,
    gemini_text: Optional[str] = None,
) -> str:
    markdown = build_markdown_report(analysis, score, plan, benchmark, scaling, gemini_text)
    if not analysis:
        body = "<p>No analysis has been generated yet.</p>"
    else:
        score_value = score.score if score else 0
        benchmark_value = "Not run"
        if benchmark and benchmark.success:
            benchmark_value = f"{benchmark.summary.max_ms:.4f} ms peak / {benchmark.summary.max_peak_memory_kb:.2f} KB peak"
        scaling_value = scaling.empirical_complexity if scaling and scaling.success else "Not run"
        findings = "".join(
            f"<li><strong>Line {item.line}: {html.escape(item.title)}</strong> - {html.escape(item.suggestion)}</li>"
            for item in analysis.line_findings
        ) or "<li>No line-level bottlenecks detected.</li>"
        patterns = "".join(
            f"<li><strong>{html.escape(item.name)}</strong> ({int(item.confidence * 100)}%) - {html.escape(item.interview_note)}</li>"
            for item in analysis.algorithm_patterns
        ) or "<li>No specific interview pattern detected.</li>"
        validation_summary = "No plan generated."
        if plan:
            validation = plan.validation
            validation_summary = (
                f"{validation.status} from {validation.source}. "
                f"Before: {validation.original_time or analysis.estimated_time} / {validation.original_space or analysis.estimated_space}; "
                f"after: {validation.candidate_time or 'Not accepted'} / {validation.candidate_space or 'Not accepted'}."
            )
        body = f"""
        <section class="grid">
          <div class="card"><span>Time Complexity</span><strong>{html.escape(analysis.estimated_time)}</strong></div>
          <div class="card"><span>Space Complexity</span><strong>{html.escape(analysis.estimated_space)}</strong></div>
          <div class="card"><span>Optimization Score</span><strong>{score_value}/100</strong></div>
          <div class="card"><span>Benchmark</span><strong>{html.escape(benchmark_value)}</strong></div>
          <div class="card"><span>Empirical Scaling</span><strong>{html.escape(scaling_value)}</strong></div>
        </section>
        <section><h2>Algorithm Patterns</h2><ul>{patterns}</ul></section>
        <section><h2>Line-Level Findings</h2><ul>{findings}</ul></section>
        <section><h2>Optimization Plan</h2><p>{html.escape(plan.summary if plan else 'No plan generated.')}</p></section>
        <section><h2>Verified Code Generation</h2><p>{html.escape(validation_summary)}</p></section>
        <section><h2>Markdown Report</h2><pre>{html.escape(markdown)}</pre></section>
        """
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Complexity Lab Report</title>
  <style>
    body {{ margin: 0; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0b1020; color: #e5e7eb; }}
    header {{ padding: 44px; background: linear-gradient(135deg, rgba(34,211,238,.28), rgba(167,139,250,.22)); border-bottom: 1px solid rgba(148,163,184,.25); }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 32px; }}
    h1 {{ margin: 0 0 10px; font-size: 42px; }}
    h2 {{ margin-top: 34px; color: #f8fafc; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin-top: 22px; }}
    .card {{ border: 1px solid rgba(148,163,184,.24); border-radius: 18px; padding: 18px; background: rgba(15,23,42,.74); }}
    .card span {{ display: block; color: #94a3b8; font-size: 12px; text-transform: uppercase; font-weight: 800; }}
    .card strong {{ display: block; margin-top: 9px; font-size: 22px; color: #f8fafc; }}
    pre {{ white-space: pre-wrap; background: #020617; color: #cbd5e1; padding: 18px; border-radius: 14px; overflow-x: auto; }}
    li {{ margin: 8px 0; }}
  </style>
</head>
<body>
  <header>
    <h1>Complexity Lab Report</h1>
    <p>Static estimates, empirical benchmarks, bottleneck evidence, and interview-ready optimization guidance.</p>
  </header>
  <main>{body}</main>
</body>
</html>
"""
