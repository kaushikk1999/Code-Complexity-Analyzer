"""Benchmark runner with timeout and traced memory measurement."""

from __future__ import annotations

import ast
import copy
import json
import math
import multiprocessing as mp
import queue
import time
import tracemalloc
from types import CodeType
from typing import Any, Dict, Iterable, List, Tuple

from benchmarking.metrics import (
    BenchmarkResult,
    BenchmarkRun,
    ScalingBenchmarkPoint,
    ScalingBenchmarkResult,
    summarize_runs,
)
from benchmarking.sandbox import build_safe_globals, validate_code_for_execution
from utils.entrypoints import (
    EntrypointDefinition,
    discover_entrypoints,
    find_entrypoint_definition,
    infer_call_from_example,
    literal_assignment_namespace,
    missing_entrypoint_message,
    validate_call_arguments,
)

SAFETY_NOTES = [
    "Benchmarks use a best-effort restricted execution context, not a secure sandbox.",
    "Static estimates and measured runtime are intentionally reported separately.",
    "Measurements reflect this machine/session and the provided benchmark input only.",
]

BENCHMARK_PROFILES = {
    "Quick": {"repeat_count": 3, "warmup_count": 1, "timeout_seconds": 3.0, "sizes": [10, 50, 100]},
    "Balanced": {"repeat_count": 5, "warmup_count": 1, "timeout_seconds": 6.0, "sizes": [10, 100, 500]},
    "Stress": {"repeat_count": 7, "warmup_count": 2, "timeout_seconds": 12.0, "sizes": [50, 500, 1500]},
}


def _parse_assignment_benchmark_input(
    text: str,
    code: str,
    definition: EntrypointDefinition,
) -> Tuple[Tuple[Any, ...], Dict[str, Any], str]:
    namespace = literal_assignment_namespace(text)
    required_args = definition.benchmark_args[: definition.required_positional_count]
    required_kwargs = definition.required_keyword_only_args
    if all(name in namespace for name in [*required_args, *required_kwargs]):
        kwargs = {name: namespace[name] for name in definition.benchmark_args if name in namespace}
        kwargs.update({name: namespace[name] for name in definition.keyword_only_args if name in namespace})
        return (), kwargs, f"assignment variable(s): {', '.join(sorted(namespace))}"
    example_args, example_kwargs, used_example = infer_call_from_example(code, definition, namespace)
    if used_example:
        return example_args, example_kwargs, "assignment variables + example call"
    kwargs = {name: namespace[name] for name in definition.benchmark_args if name in namespace}
    return (), kwargs, f"assignment variable(s): {', '.join(sorted(namespace))}"


def _parse_benchmark_input(
    input_text: str,
    code: str = "",
    definition: EntrypointDefinition = None,
) -> Tuple[Tuple[Any, ...], Dict[str, Any], str]:
    text = (input_text or "").strip()
    if not text:
        return (), {}, "no arguments"

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            value = ast.literal_eval(text)
        except Exception as exc:
            if definition:
                try:
                    return _parse_assignment_benchmark_input(text, code, definition)
                except Exception:
                    pass
            raise ValueError(f"Benchmark input must be JSON or a Python literal: {exc}") from exc

    if isinstance(value, dict) and ("args" in value or "kwargs" in value):
        args_value = value.get("args", [])
        kwargs_value = value.get("kwargs", {})
        if not isinstance(args_value, (list, tuple)):
            raise ValueError("`args` must be a list or tuple.")
        if not isinstance(kwargs_value, dict):
            raise ValueError("`kwargs` must be an object/dict.")
        return tuple(args_value), kwargs_value, f"{len(args_value)} positional arg(s), {len(kwargs_value)} keyword arg(s)"

    return (value,), {}, "single positional argument"


def _clone_inputs(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    try:
        return copy.deepcopy(args), copy.deepcopy(kwargs)
    except Exception:
        return args, dict(kwargs)


def _execute_user_code(compiled: CodeType) -> Dict[str, Any]:
    namespace = build_safe_globals()
    exec(compiled, namespace, namespace)
    return namespace


def _resolve_entrypoint(namespace: Dict[str, Any], entrypoint: str) -> Any:
    if "." in entrypoint:
        class_name, method_name = entrypoint.split(".", 1)
        class_value = namespace.get(class_name)
        if isinstance(class_value, type):
            raw_method = class_value.__dict__.get(method_name)
            if isinstance(raw_method, (staticmethod, classmethod)):
                method = getattr(class_value, method_name, None)
                if callable(method):
                    return method
            try:
                instance = class_value()
            except Exception:
                instance = None
            if instance is not None:
                method = getattr(instance, method_name, None)
                if callable(method):
                    return method
            method = getattr(class_value, method_name, None)
            if callable(method):
                return method
        return None

    target = namespace.get(entrypoint)
    if callable(target):
        return target
    for value in namespace.values():
        if not isinstance(value, type):
            continue
        raw_method = value.__dict__.get(entrypoint)
        if isinstance(raw_method, (staticmethod, classmethod)):
            method = getattr(value, entrypoint, None)
            if callable(method):
                return method
        try:
            instance = value()
        except Exception:
            continue
        method = getattr(instance, entrypoint, None)
        if callable(method):
            return method
    return None


def _worker(
    code: str,
    entrypoint: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    repeat_count: int,
    warmup_count: int,
    allow_top_level: bool,
    result_queue: "mp.Queue[Dict[str, Any]]",
) -> None:
    try:
        compiled = compile(code, "<user_code>", "exec")
        namespace = _execute_user_code(compiled)

        runs: List[Dict[str, float]] = []
        if entrypoint:
            target = _resolve_entrypoint(namespace, entrypoint)
            if not callable(target):
                raise ValueError(f"Entrypoint `{entrypoint}` was not found or is not callable.")

            for _ in range(max(0, warmup_count)):
                run_args, run_kwargs = _clone_inputs(args, kwargs)
                target(*run_args, **run_kwargs)

            for index in range(repeat_count):
                run_args, run_kwargs = _clone_inputs(args, kwargs)
                tracemalloc.start()
                start = time.perf_counter_ns()
                target(*run_args, **run_kwargs)
                end = time.perf_counter_ns()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                runs.append(
                    {
                        "run_index": index + 1,
                        "runtime_ms": (end - start) / 1_000_000,
                        "current_memory_kb": current / 1024,
                        "peak_memory_kb": peak / 1024,
                    }
                )
        else:
            if not allow_top_level:
                raise ValueError(
                    "Top-level script benchmarking is disabled. Provide an entrypoint or enable top-level execution."
                )
            for _ in range(max(0, warmup_count)):
                _execute_user_code(compiled)
            for index in range(repeat_count):
                tracemalloc.start()
                start = time.perf_counter_ns()
                _execute_user_code(compiled)
                end = time.perf_counter_ns()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                runs.append(
                    {
                        "run_index": index + 1,
                        "runtime_ms": (end - start) / 1_000_000,
                        "current_memory_kb": current / 1024,
                        "peak_memory_kb": peak / 1024,
                    }
                )
        result_queue.put({"success": True, "runs": runs})
    except Exception as exc:
        try:
            tracemalloc.stop()
        except RuntimeError:
            pass
        result_queue.put({"success": False, "error": f"{type(exc).__name__}: {exc}"})


def run_benchmark(
    code: str,
    entrypoint: str = "",
    input_text: str = "",
    repeat_count: int = 5,
    warmup_count: int = 1,
    timeout_seconds: float = 5.0,
    allow_top_level: bool = False,
) -> BenchmarkResult:
    entrypoint = (entrypoint or "").strip()
    repeat_count = max(1, min(int(repeat_count or 1), 30))
    warmup_count = max(0, min(int(warmup_count or 0), 10))
    timeout_seconds = max(1.0, min(float(timeout_seconds or 5.0), 20.0))

    violations = validate_code_for_execution(code)
    if violations:
        return BenchmarkResult(
            success=False,
            entrypoint=entrypoint or "<module>",
            input_description="not executed",
            error="Benchmark blocked by safety checks: " + " ".join(violations[:5]),
            safety_notes=SAFETY_NOTES,
        )

    definitions = discover_entrypoints(code)
    definition = find_entrypoint_definition(definitions, entrypoint)
    if entrypoint and not definition:
        return BenchmarkResult(
            success=False,
            entrypoint=entrypoint,
            input_description="not executed",
            error=missing_entrypoint_message(entrypoint, definitions),
            safety_notes=SAFETY_NOTES,
        )

    try:
        args, kwargs, input_description = _parse_benchmark_input(input_text, code, definition)
    except ValueError as exc:
        return BenchmarkResult(
            success=False,
            entrypoint=entrypoint or "<module>",
            input_description="invalid input",
            error=str(exc),
            safety_notes=SAFETY_NOTES,
        )

    if definition:
        argument_error = validate_call_arguments(definition, args, kwargs)
        if argument_error:
            return BenchmarkResult(
                success=False,
                entrypoint=entrypoint,
                input_description=input_description,
                error=argument_error,
                safety_notes=SAFETY_NOTES,
            )
        entrypoint = definition.callable_name

    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    context: Any = mp.get_context(start_method)
    result_queue: "mp.Queue[Dict[str, Any]]" = context.Queue()
    process = context.Process(
        target=_worker,
        args=(code, entrypoint, args, kwargs, repeat_count, warmup_count, allow_top_level, result_queue),
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(1)
        return BenchmarkResult(
            success=False,
            entrypoint=entrypoint or "<module>",
            input_description=input_description,
            error=f"Benchmark timed out after {timeout_seconds:.1f}s.",
            safety_notes=SAFETY_NOTES,
        )

    try:
        payload = result_queue.get_nowait()
    except queue.Empty:
        return BenchmarkResult(
            success=False,
            entrypoint=entrypoint or "<module>",
            input_description=input_description,
            error="Benchmark worker exited without returning a result.",
            safety_notes=SAFETY_NOTES,
        )

    if not payload.get("success"):
        return BenchmarkResult(
            success=False,
            entrypoint=entrypoint or "<module>",
            input_description=input_description,
            error=payload.get("error", "Unknown benchmark error."),
            safety_notes=SAFETY_NOTES,
        )

    runs = [BenchmarkRun(**item) for item in payload.get("runs", [])]
    return BenchmarkResult(
        success=True,
        entrypoint=entrypoint or "<module>",
        input_description=input_description,
        runs=runs,
        summary=summarize_runs(runs),
        safety_notes=SAFETY_NOTES,
    )


def profile_settings(profile: str) -> Dict[str, Any]:
    return dict(BENCHMARK_PROFILES.get(profile, BENCHMARK_PROFILES["Balanced"]))


def should_run_auto_benchmark(static_only_mode: bool) -> bool:
    return not static_only_mode


def _resize_value(value: Any, size: int, shape: str) -> Any:
    shape = shape.lower()
    if shape == "string":
        return "".join(chr(97 + (i % 26)) for i in range(size))
    if shape == "matrix":
        width = max(1, int(math.sqrt(size)))
        return [[(row * width + col) % 997 for col in range(width)] for row in range(width)]
    if shape == "dict":
        return {f"k{i}": i for i in range(size)}
    if shape == "graph":
        return {i: [j for j in (i + 1, i + 2) if j < size] for i in range(size)}
    if isinstance(value, str):
        return _resize_value(value, size, "string")
    if isinstance(value, dict):
        return _resize_value(value, size, "dict")
    if isinstance(value, list) and value and isinstance(value[0], list):
        return _resize_value(value, size, "matrix")
    if isinstance(value, (list, tuple, set)):
        return list(range(size))
    return list(range(size))


def build_scaled_input(input_text: str, size: int, data_shape: str) -> str:
    args, kwargs, _ = _parse_benchmark_input(input_text)
    if args:
        resized_args = list(args)
        resized_args[0] = _resize_value(args[0], size, data_shape)
        return json.dumps({"args": resized_args, "kwargs": kwargs})
    if kwargs:
        first_key = next(iter(kwargs))
        resized_kwargs = dict(kwargs)
        resized_kwargs[first_key] = _resize_value(kwargs[first_key], size, data_shape)
        return json.dumps({"args": [], "kwargs": resized_kwargs})
    return json.dumps({"args": [_resize_value([], size, data_shape)], "kwargs": {}})


def _r_squared(actual: List[float], predicted: List[float]) -> float:
    if len(actual) < 2:
        return 0.0
    mean_actual = sum(actual) / len(actual)
    total = sum((value - mean_actual) ** 2 for value in actual)
    residual = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    if total <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, 1 - residual / total))


def _fit_model(sizes: List[int], times: List[float], transform: Iterable[float]) -> float:
    x = list(transform)
    y = times
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    denom = sum((item - mean_x) ** 2 for item in x)
    if denom <= 1e-12:
        return 0.0
    slope = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / denom
    intercept = mean_y - slope * mean_x
    predicted = [intercept + slope * xi for xi in x]
    return round(_r_squared(y, predicted), 4)


def estimate_empirical_complexity(points: List[ScalingBenchmarkPoint]) -> Tuple[str, Dict[str, float]]:
    good = [point for point in points if point.success and point.avg_ms > 0]
    if len(good) < 3:
        return "Unknown", {}
    sizes = [point.input_size for point in good]
    times = [point.avg_ms for point in good]
    fit_scores = {
        "O(1)": _fit_model(sizes, times, [1 for _ in sizes]),
        "O(log n)": _fit_model(sizes, times, [math.log2(max(2, n)) for n in sizes]),
        "O(n)": _fit_model(sizes, times, [n for n in sizes]),
        "O(n log n)": _fit_model(sizes, times, [n * math.log2(max(2, n)) for n in sizes]),
        "O(n^2)": _fit_model(sizes, times, [n * n for n in sizes]),
    }
    winner = max(fit_scores, key=fit_scores.get)
    return winner, fit_scores


def run_scaling_benchmark(
    code: str,
    entrypoint: str,
    input_text: str,
    data_shape: str = "list",
    sizes: List[int] = None,
    repeat_count: int = 3,
    warmup_count: int = 1,
    timeout_seconds: float = 6.0,
) -> ScalingBenchmarkResult:
    entrypoint = (entrypoint or "").strip()
    if not entrypoint:
        return ScalingBenchmarkResult(
            success=False,
            entrypoint="<module>",
            data_shape=data_shape,
            error="Scaling benchmark requires an entrypoint function.",
        )

    sizes = sizes or [10, 100, 500]
    points: List[ScalingBenchmarkPoint] = []
    for size in sizes:
        try:
            scaled_input = build_scaled_input(input_text, int(size), data_shape)
        except Exception as exc:
            points.append(
                ScalingBenchmarkPoint(
                    input_size=int(size),
                    success=False,
                    error=f"Could not build scaled input: {type(exc).__name__}: {exc}",
                )
            )
            continue
        result = run_benchmark(
            code=code,
            entrypoint=entrypoint,
            input_text=scaled_input,
            repeat_count=repeat_count,
            warmup_count=warmup_count,
            timeout_seconds=timeout_seconds,
            allow_top_level=False,
        )
        if result.success:
            points.append(
                ScalingBenchmarkPoint(
                    input_size=int(size),
                    success=True,
                    avg_ms=result.summary.avg_ms,
                    min_ms=result.summary.min_ms,
                    max_ms=result.summary.max_ms,
                    max_peak_memory_kb=result.summary.max_peak_memory_kb,
                )
            )
        else:
            points.append(
                ScalingBenchmarkPoint(
                    input_size=int(size),
                    success=False,
                    error=result.error,
                )
            )

    empirical, fit_scores = estimate_empirical_complexity(points)
    return ScalingBenchmarkResult(
        success=any(point.success for point in points),
        entrypoint=entrypoint,
        data_shape=data_shape,
        empirical_complexity=empirical,
        fit_scores=fit_scores,
        points=points,
        error=None if any(point.success for point in points) else "No scaling benchmark point completed successfully.",
    )
