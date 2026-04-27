"""Transparent optimization scoring model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from analyzer.complexity_rules import complexity_rank
from analyzer.models import StaticAnalysisResult
from benchmarking.metrics import BenchmarkResult


@dataclass
class ScoreBreakdown:
    score: int
    efficiency_percentage: int
    severity: str
    improvement_potential: str
    components: Dict[str, float] = field(default_factory=dict)
    penalties: List[str] = field(default_factory=list)
    bottlenecks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _severity(score: int) -> str:
    if score >= 86:
        return "excellent"
    if score >= 72:
        return "strong"
    if score >= 55:
        return "moderate"
    if score >= 38:
        return "risky"
    return "critical"


def _improvement_potential(score: int) -> str:
    if score >= 86:
        return "Low"
    if score >= 72:
        return "Moderate"
    if score >= 55:
        return "Meaningful"
    return "High"


def _bounded(value: float, maximum: float) -> float:
    return round(max(0.0, min(maximum, value)), 2)


def calculate_optimization_score(
    analysis: StaticAnalysisResult,
    benchmark: Optional[BenchmarkResult] = None,
) -> ScoreBreakdown:
    metrics = analysis.metrics or {}
    penalties: List[str] = []

    algorithmic = 30.0
    time_rank = complexity_rank(analysis.estimated_time)
    if time_rank >= complexity_rank("O(2^n)"):
        algorithmic -= 24
        penalties.append("Very high estimated time complexity.")
    elif time_rank >= complexity_rank("O(n^3)"):
        algorithmic -= 18
        penalties.append("Cubic-style loop structure detected.")
    elif time_rank >= complexity_rank("O(n^2)"):
        algorithmic -= 12
        penalties.append("Quadratic-style loop structure detected.")
    elif time_rank >= complexity_rank("O(n log n)"):
        algorithmic -= 4
        penalties.append("Sorting or n log n work is present.")

    nested_loop = 18.0 - min(18.0, 7.0 * max(0, metrics.get("max_loop_depth", 0) - 1))
    if metrics.get("max_loop_depth", 0) >= 2:
        penalties.append("Nested loops reduce scalability.")

    data_structure = 14.0
    data_structure -= min(10.0, 5.0 * metrics.get("list_membership_in_loop", 0))
    data_structure -= min(4.0, 2.0 * metrics.get("membership_tests", 0))
    if metrics.get("list_membership_in_loop", 0):
        penalties.append("List-style membership checks should often become set/dict lookups.")

    redundant = 12.0
    redundant -= min(8.0, 6.0 * metrics.get("repeated_expensive_calls", 0))
    redundant -= min(5.0, 2.5 * metrics.get("repeated_traversals", 0))
    redundant -= min(5.0, 5.0 * metrics.get("sort_in_loop", 0))
    if metrics.get("repeated_expensive_calls", 0) or metrics.get("sort_in_loop", 0):
        penalties.append("Redundant expensive work is present.")

    memory = 10.0
    space_rank = complexity_rank(analysis.estimated_space)
    if space_rank >= complexity_rank("O(n^2)"):
        memory -= 7
        penalties.append("Estimated auxiliary memory may grow quadratically.")
    elif space_rank >= complexity_rank("O(n)"):
        memory -= 2
    memory -= min(4.0, 1.2 * metrics.get("temp_objects", 0))
    memory -= min(3.0, 2.0 * metrics.get("slicing_in_loop", 0))

    maintainability = 10.0
    maintainability -= min(3.0, 1.0 * metrics.get("while_loops", 0))
    maintainability -= min(3.0, 1.0 * metrics.get("recursive_calls", 0))
    maintainability -= 2.0 if analysis.confidence < 0.55 else 0.0
    maintainability -= min(2.0, max(0, len(analysis.caveats) - 2) * 0.5)

    benchmark_signal = 6.0
    if benchmark and benchmark.success and benchmark.summary.repeat_count:
        avg_ms = benchmark.summary.avg_ms
        peak_kb = benchmark.summary.max_peak_memory_kb
        if avg_ms > 1000:
            benchmark_signal -= 5
            penalties.append("Empirical runtime is high for the supplied input.")
        elif avg_ms > 200:
            benchmark_signal -= 3
            penalties.append("Empirical runtime may be noticeable for interactive use.")
        elif avg_ms > 50:
            benchmark_signal -= 1
        if peak_kb > 50_000:
            benchmark_signal -= 2
            penalties.append("Peak traced Python memory is high for the supplied input.")
    elif benchmark and not benchmark.success:
        benchmark_signal -= 2
        penalties.append("Benchmark did not complete, so empirical confidence is lower.")

    components = {
        "Algorithmic efficiency": _bounded(algorithmic, 30.0),
        "Nested-loop burden": _bounded(nested_loop, 18.0),
        "Data structure fit": _bounded(data_structure, 14.0),
        "Redundant work": _bounded(redundant, 12.0),
        "Memory discipline": _bounded(memory, 10.0),
        "Maintainability": _bounded(maintainability, 10.0),
        "Benchmark signal": _bounded(benchmark_signal, 6.0),
    }
    score = int(round(sum(components.values())))
    bottlenecks = [pattern.name for pattern in analysis.anti_patterns[:5]]
    return ScoreBreakdown(
        score=max(0, min(100, score)),
        efficiency_percentage=max(0, min(100, score)),
        severity=_severity(score),
        improvement_potential=_improvement_potential(score),
        components=components,
        penalties=list(dict.fromkeys(penalties)),
        bottlenecks=bottlenecks,
    )
