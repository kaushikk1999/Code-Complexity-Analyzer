"""Benchmark metric models and summary helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkRun:
    run_index: int
    runtime_ms: float
    current_memory_kb: float
    peak_memory_kb: float


@dataclass
class BenchmarkSummary:
    repeat_count: int = 0
    min_ms: float = 0.0
    avg_ms: float = 0.0
    max_ms: float = 0.0
    std_ms: float = 0.0
    avg_current_memory_kb: float = 0.0
    avg_peak_memory_kb: float = 0.0
    max_peak_memory_kb: float = 0.0


@dataclass
class BenchmarkResult:
    success: bool
    entrypoint: str
    input_description: str
    runs: List[BenchmarkRun] = field(default_factory=list)
    summary: BenchmarkSummary = field(default_factory=BenchmarkSummary)
    error: Optional[str] = None
    safety_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScalingBenchmarkPoint:
    input_size: int
    success: bool
    avg_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    max_peak_memory_kb: float = 0.0
    error: Optional[str] = None


@dataclass
class ScalingBenchmarkResult:
    success: bool
    entrypoint: str
    data_shape: str
    empirical_complexity: str = "Unknown"
    fit_scores: Dict[str, float] = field(default_factory=dict)
    points: List[ScalingBenchmarkPoint] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize_runs(runs: List[BenchmarkRun]) -> BenchmarkSummary:
    if not runs:
        return BenchmarkSummary()
    runtimes = [run.runtime_ms for run in runs]
    current = [run.current_memory_kb for run in runs]
    peaks = [run.peak_memory_kb for run in runs]
    return BenchmarkSummary(
        repeat_count=len(runs),
        min_ms=round(min(runtimes), 4),
        avg_ms=round(mean(runtimes), 4),
        max_ms=round(max(runtimes), 4),
        std_ms=round(pstdev(runtimes), 4) if len(runtimes) > 1 else 0.0,
        avg_current_memory_kb=round(mean(current), 4),
        avg_peak_memory_kb=round(mean(peaks), 4),
        max_peak_memory_kb=round(max(peaks), 4),
    )
