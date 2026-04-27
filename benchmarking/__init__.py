"""Benchmarking package for safe-ish code execution and profiling."""

from benchmarking.docker_backend import docker_available, run_benchmark_in_docker
from benchmarking.metrics import (
    BenchmarkResult,
    BenchmarkRun,
    BenchmarkSummary,
    ScalingBenchmarkPoint,
    ScalingBenchmarkResult,
)
from benchmarking.runner import (
    profile_settings,
    run_benchmark,
    run_scaling_benchmark,
    should_run_auto_benchmark,
)

__all__ = [
    "BenchmarkResult",
    "BenchmarkRun",
    "BenchmarkSummary",
    "ScalingBenchmarkPoint",
    "ScalingBenchmarkResult",
    "run_benchmark",
    "run_scaling_benchmark",
    "profile_settings",
    "should_run_auto_benchmark",
    "docker_available",
    "run_benchmark_in_docker",
]
