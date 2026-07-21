import threading

from analyzer import analyze_code
from optimization import (
    OptimizedCodeCandidate,
    generate_multi_model_verified_candidates,
)
from scoring import calculate_optimization_score

ORIGINAL = "def two_sum(nums, target):\n    for i in range(len(nums)):\n        for j in range(i + 1, len(nums)):\n            if nums[i] + nums[j] == target:\n                return [i, j]\n    return []\n"
FAST = "def two_sum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        if target - num in seen:\n            return [seen[target - num], i]\n        seen[num] = i\n    return []\n"
BENCH_INPUT = '{"kwargs": {"nums": [2, 7, 11, 15], "target": 9}}'


def _fixtures():
    analysis = analyze_code(ORIGINAL)
    return analysis, calculate_optimization_score(analysis)


def test_all_models_are_asked_and_generation_runs_in_parallel():
    analysis, score = _fixtures()
    models = ["model-a", "model-b", "model-c", "model-d"]
    seen_threads = set()
    calls = []
    lock = threading.Lock()

    def provider(model, level, rejection_reasons):
        with lock:
            calls.append((model, level))
            seen_threads.add(threading.get_ident())
        return OptimizedCodeCandidate(source="ollama", code=FAST, explanation="hash map", level=level), None

    plan = generate_multi_model_verified_candidates(
        original_code=ORIGINAL,
        analysis=analysis,
        score=score,
        entrypoint="two_sum",
        benchmark_input=BENCH_INPUT,
        models=models,
        model_candidate_provider=provider,
        repeat_count=1,
        timeout_seconds=5.0,
    )

    assert {model for model, _ in calls} == set(models)
    assert len(calls) == len(models) * 3
    # Generation is fanned out, so more than one worker thread must have run.
    assert len(seen_threads) > 1
    assert {row.model for row in plan.model_comparison} == set(models)


def test_winner_is_the_lowest_measured_runtime_and_is_flagged_once():
    analysis, score = _fixtures()

    def provider(model, level, rejection_reasons):
        code = FAST if model == "fast-model" else ORIGINAL
        return OptimizedCodeCandidate(source="ollama", code=code, explanation="x", level=level), None

    plan = generate_multi_model_verified_candidates(
        original_code=ORIGINAL,
        analysis=analysis,
        score=score,
        entrypoint="two_sum",
        benchmark_input=BENCH_INPUT,
        models=["slow-model", "fast-model"],
        model_candidate_provider=provider,
        repeat_count=1,
        timeout_seconds=5.0,
    )

    accepted = [row for row in plan.model_comparison if row.status != "rejected"]
    assert accepted, "expected at least one accepted candidate"
    # Rows are ranked by measured runtime, then peak memory.
    runtimes = [row.benchmark_avg_ms for row in accepted]
    assert runtimes == sorted(runtimes)
    assert sum(1 for row in plan.model_comparison if row.is_winner) <= 1


def test_a_failing_model_does_not_kill_the_batch():
    analysis, score = _fixtures()

    def provider(model, level, rejection_reasons):
        if model == "broken":
            raise RuntimeError("boom")
        return OptimizedCodeCandidate(source="ollama", code=FAST, explanation="x", level=level), None

    plan = generate_multi_model_verified_candidates(
        original_code=ORIGINAL,
        analysis=analysis,
        score=score,
        entrypoint="two_sum",
        benchmark_input=BENCH_INPUT,
        models=["broken", "working"],
        model_candidate_provider=provider,
        repeat_count=1,
        timeout_seconds=5.0,
    )

    assert plan.best_candidate is not None
    assert plan.best_candidate.model == "working"
    assert any("broken" in note for note in plan.generation_notes)


def test_falls_back_to_local_when_every_model_returns_nothing():
    analysis, score = _fixtures()

    def provider(model, level, rejection_reasons):
        return None, "model unavailable"

    plan = generate_multi_model_verified_candidates(
        original_code=ORIGINAL,
        analysis=analysis,
        score=score,
        entrypoint="two_sum",
        benchmark_input=BENCH_INPUT,
        models=["a", "b"],
        model_candidate_provider=provider,
        repeat_count=1,
        timeout_seconds=5.0,
    )

    assert all(candidate.model == "local" for candidate in plan.verified_candidates)
