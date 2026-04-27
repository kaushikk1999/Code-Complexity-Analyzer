from analyzer import analyze_code
from benchmarking.metrics import BenchmarkResult, BenchmarkRun, summarize_runs
from scoring import calculate_optimization_score


def test_simple_solution_scores_high():
    analysis = analyze_code(
        """
def total(values):
    return sum(values)
"""
    )
    score = calculate_optimization_score(analysis)
    assert 70 <= score.score <= 100
    assert score.efficiency_percentage == score.score


def test_nested_loop_penalty():
    analysis = analyze_code(
        """
def brute(values):
    count = 0
    for a in values:
        for b in values:
            if a == b:
                count += 1
    return count
"""
    )
    score = calculate_optimization_score(analysis)
    assert score.score < 80
    assert any("Nested" in bottleneck for bottleneck in score.bottlenecks)


def test_redundant_work_penalty():
    analysis = analyze_code(
        """
def repeated(values):
    return sum(values) + sum(values) + max(values)
"""
    )
    score = calculate_optimization_score(analysis)
    assert score.score < 95
    assert any("Redundant" in penalty or "repeated" in penalty.lower() for penalty in score.penalties)


def test_benchmark_signal_penalty():
    analysis = analyze_code(
        """
def identity(values):
    return values
"""
    )
    runs = [
        BenchmarkRun(run_index=1, runtime_ms=1200.0, current_memory_kb=1.0, peak_memory_kb=60_000.0),
        BenchmarkRun(run_index=2, runtime_ms=1100.0, current_memory_kb=1.0, peak_memory_kb=55_000.0),
    ]
    benchmark = BenchmarkResult(
        success=True,
        entrypoint="identity",
        input_description="test",
        runs=runs,
        summary=summarize_runs(runs),
    )
    score = calculate_optimization_score(analysis, benchmark)
    assert score.components["Benchmark signal"] < 6
    assert score.score <= 94


def test_score_bounds_are_deterministic():
    code = """
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
"""
    first = calculate_optimization_score(analyze_code(code))
    second = calculate_optimization_score(analyze_code(code))
    assert first.score == second.score
    assert 0 <= first.score <= 100
