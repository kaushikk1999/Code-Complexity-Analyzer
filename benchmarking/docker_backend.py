"""Optional Docker-based benchmark isolation.

This backend is intentionally optional. Streamlit Community Cloud generally will
not expose Docker, but local/professional deployments can enable it for stronger
process, filesystem, memory, CPU, PID, and network isolation.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

from benchmarking.metrics import BenchmarkResult, BenchmarkRun, summarize_runs
from utils.entrypoints import discover_entrypoints, find_entrypoint_definition, missing_entrypoint_message


@dataclass
class DockerBenchmarkConfig:
    image: str = "python:3.11-slim"
    memory: str = "256m"
    cpus: str = "1.0"
    pids_limit: int = 64


def docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        result = subprocess.run(["docker", "version"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def run_benchmark_in_docker(
    code: str,
    entrypoint: str,
    input_text: str,
    repeat_count: int,
    warmup_count: int,
    timeout_seconds: float,
    config: DockerBenchmarkConfig = None,
) -> BenchmarkResult:
    config = config or DockerBenchmarkConfig()
    if not docker_available():
        return BenchmarkResult(
            success=False,
            entrypoint=entrypoint or "<module>",
            input_description="docker unavailable",
            error="Docker is not available on this machine or daemon access is denied.",
            safety_notes=["Docker backend was requested but could not start."],
        )
    if not entrypoint:
        return BenchmarkResult(
            success=False,
            entrypoint="<module>",
            input_description="not executed",
            error="Docker backend requires an entrypoint function.",
            safety_notes=["Top-level script execution is disabled in Docker backend."],
        )

    definitions = discover_entrypoints(code)
    definition = find_entrypoint_definition(definitions, entrypoint)
    if not definition:
        return BenchmarkResult(
            success=False,
            entrypoint=entrypoint,
            input_description="not executed",
            error=missing_entrypoint_message(entrypoint, definitions),
            safety_notes=["Docker backend was requested, but the configured entrypoint was not found."],
        )
    entrypoint = definition.callable_name

    runner_source = textwrap.dedent(
        f"""
        import ast, copy, json, time, tracemalloc, statistics

        CODE = {json.dumps(code)}
        ENTRYPOINT = {json.dumps(entrypoint)}
        INPUT_TEXT = {json.dumps(input_text)}
        REPEAT_COUNT = {int(repeat_count)}
        WARMUP_COUNT = {int(warmup_count)}

        def parse_input(text):
            text = (text or "").strip()
            if not text:
                return (), {{}}, "no arguments"
            try:
                value = json.loads(text)
            except Exception:
                value = ast.literal_eval(text)
            if isinstance(value, dict) and ("args" in value or "kwargs" in value):
                return tuple(value.get("args", [])), value.get("kwargs", {{}}), "structured"
            return (value,), {{}}, "single positional"

        def clone(args, kwargs):
            try:
                return copy.deepcopy(args), copy.deepcopy(kwargs)
            except Exception:
                return args, dict(kwargs)

        try:
            args, kwargs, input_description = parse_input(INPUT_TEXT)
            namespace = {{}}
            exec(compile(CODE, "<user_code>", "exec"), namespace, namespace)
            if "." in ENTRYPOINT:
                class_name, method_name = ENTRYPOINT.split(".", 1)
                class_value = namespace.get(class_name)
                target = None
                if isinstance(class_value, type):
                    raw_method = class_value.__dict__.get(method_name)
                    if isinstance(raw_method, (staticmethod, classmethod)):
                        target = getattr(class_value, method_name, None)
                    if target is None:
                        try:
                            target = getattr(class_value(), method_name, None)
                        except Exception:
                            target = getattr(class_value, method_name, None)
            else:
                target = namespace.get(ENTRYPOINT)
                if target is None:
                    for value in namespace.values():
                        if not isinstance(value, type):
                            continue
                        raw_method = value.__dict__.get(ENTRYPOINT)
                        if isinstance(raw_method, (staticmethod, classmethod)):
                            target = getattr(value, ENTRYPOINT, None)
                            break
                        try:
                            target = getattr(value(), ENTRYPOINT, None)
                        except Exception:
                            continue
                        if callable(target):
                            break
            if not callable(target):
                raise ValueError(f"Entrypoint {{ENTRYPOINT}} was not found or callable.")
            for _ in range(max(0, WARMUP_COUNT)):
                a, kw = clone(args, kwargs)
                target(*a, **kw)
            runs = []
            for index in range(REPEAT_COUNT):
                a, kw = clone(args, kwargs)
                tracemalloc.start()
                start = time.perf_counter_ns()
                target(*a, **kw)
                end = time.perf_counter_ns()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                runs.append({{
                    "run_index": index + 1,
                    "runtime_ms": (end - start) / 1_000_000,
                    "current_memory_kb": current / 1024,
                    "peak_memory_kb": peak / 1024,
                }})
            print(json.dumps({{"success": True, "input_description": input_description, "runs": runs}}))
        except Exception as exc:
            print(json.dumps({{"success": False, "error": f"{{type(exc).__name__}}: {{exc}}"}}))
        """
    )
    with tempfile.TemporaryDirectory(prefix="complexity-lab-docker-") as tmp:
        runner_path = Path(tmp) / "runner.py"
        runner_path.write_text(runner_source, encoding="utf-8")
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--memory",
            config.memory,
            "--cpus",
            config.cpus,
            "--pids-limit",
            str(config.pids_limit),
            "-v",
            f"{tmp}:/work:ro",
            "-w",
            "/work",
            config.image,
            "python",
            "runner.py",
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds + 8)
        except subprocess.TimeoutExpired:
            return BenchmarkResult(
                success=False,
                entrypoint=entrypoint,
                input_description="docker timeout",
                error=f"Docker benchmark timed out after {timeout_seconds + 8:.1f}s.",
                safety_notes=["Docker backend timed out."],
            )
    if completed.returncode != 0:
        return BenchmarkResult(
            success=False,
            entrypoint=entrypoint,
            input_description="docker failure",
            error=(completed.stderr or completed.stdout).strip()[:1000],
            safety_notes=["Docker process failed before returning benchmark JSON."],
        )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return BenchmarkResult(
            success=False,
            entrypoint=entrypoint,
            input_description="docker parse failure",
            error=f"Could not parse Docker benchmark output: {type(exc).__name__}: {exc}",
            safety_notes=["Docker returned non-JSON output."],
        )
    if not payload.get("success"):
        return BenchmarkResult(
            success=False,
            entrypoint=entrypoint,
            input_description="docker execution",
            error=payload.get("error", "Unknown Docker benchmark failure."),
            safety_notes=["Docker backend executed but user code failed."],
        )
    runs = [BenchmarkRun(**item) for item in payload.get("runs", [])]
    return BenchmarkResult(
        success=True,
        entrypoint=entrypoint,
        input_description=payload.get("input_description", "docker"),
        runs=runs,
        summary=summarize_runs(runs),
        safety_notes=[
            "Benchmark executed in Docker with no network, memory quota, CPU quota, read-only mounted runner, and PID limit.",
            "Docker improves isolation but still should be used only on infrastructure you control.",
        ],
    )
