"""Benchmark endpoint: execute a Python snippet across 30 input sizes.

Measures wall-clock time and peak memory per size and returns points for a
time-vs-size graph, plus a rough fitted growth curve.

SECURITY / PLATFORM NOTE: this executes user code. It runs fine locally
(a real Python process). On Vercel's serverless runtime it is best-effort:
short timeouts and an ephemeral sandbox mean heavy inputs may be cut off.
A small denylist blocks the most dangerous stdlib calls.
"""

from __future__ import annotations

import ast
import inspect
import json
import random
import time
import tracemalloc
from http.server import BaseHTTPRequestHandler

N_POINTS = 30
PER_POINT_BUDGET = 1.2      # seconds; stop scaling once one point exceeds this
TOTAL_BUDGET = 25.0         # seconds; hard cap on the whole run

DENY = ("import os", "import sys", "import subprocess", "import socket",
        "__import__", "eval(", "exec(", "open(", "compile(", "input(",
        "os.system", "shutil", "pathlib", "requests", "urllib")

LIST_NAMES = {"nums", "arr", "array", "values", "vals", "items", "data",
              "a", "lst", "list", "scores", "seq", "elements", "xs"}
INT_NAMES = {"target", "k", "n", "value", "count", "size", "limit", "num", "width"}


def _guard(code: str) -> str:
    low = code.lower()
    for bad in DENY:
        if bad in low:
            return f"Refused: benchmark sandbox blocks '{bad.strip()}'."
    return ""


def _first_function(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    return None


def _make_arg(name: str, size: int):
    n = name.lower()
    if n in INT_NAMES:
        return max(1, min(size, 30)) if n == "n" else max(1, size // 2)
    if "matrix" in n or "grid" in n:
        side = max(1, int(size ** 0.5))
        return [[random.randint(0, size) for _ in range(side)] for _ in range(side)]
    if "graph" in n:
        return {str(i): [str((i + 1) % size)] for i in range(max(1, size))}
    if "str" in n or "text" in n or "word" in n:
        return "".join(random.choice("abcde") for _ in range(size))
    # default: a list of ints of length `size`
    return [random.randint(0, size) for _ in range(size)]


def _build_args(func, size: int):
    sig = inspect.signature(func)
    args = []
    for i, (pname, p) in enumerate(sig.parameters.items()):
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is not inspect.Parameter.empty and i > 1:
            continue  # let optional trailing args default
        args.append(_make_arg(pname if pname else f"p{i}", size))
    return args


def _run(raw_body: bytes) -> dict:
    try:
        data = json.loads(raw_body or b"{}")
    except (ValueError, TypeError):
        data = {}
    code = (data.get("code") or "").strip()
    if not code:
        return {"ok": False, "error": "No code provided."}

    guard = _guard(code)
    if guard:
        return {"ok": False, "error": guard}

    entry = (data.get("entrypoint") or "").strip() or _first_function(code)
    if not entry:
        return {"ok": False, "error": "No top-level function found to benchmark."}

    ns: dict = {}
    try:
        exec(compile(code, "<snippet>", "exec"), ns)  # noqa: S102 (intentional)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Code failed to load: {exc}"}
    func = ns.get(entry)
    if not callable(func):
        return {"ok": False, "error": f"'{entry}' is not callable."}

    # 30 increasing input sizes.
    sizes = [ (i + 1) * 20 for i in range(N_POINTS) ]
    points = []
    started = time.perf_counter()
    for size in sizes:
        try:
            args = _build_args(func, size)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"Could not build inputs: {exc}"}
        # time: best of 3 short repeats (wall clock + CPU time)
        best = None
        best_cpu = None
        tracemalloc.start()
        try:
            for _ in range(3):
                t0, c0 = time.perf_counter(), time.process_time()
                func(*args)
                dt = time.perf_counter() - t0
                dc = time.process_time() - c0
                if best is None or dt < best:
                    best, best_cpu = dt, dc
        except Exception as exc:  # noqa: BLE001
            tracemalloc.stop()
            return {"ok": False, "error": f"Call failed at size {size}: {exc}"}
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        points.append({"n": size, "ms": round(best * 1000, 4),
                       "cpu_ms": round(best_cpu * 1000, 4), "kb": round(peak / 1024, 1)})
        if best > PER_POINT_BUDGET or (time.perf_counter() - started) > TOTAL_BUDGET:
            break

    total_ms = round((time.perf_counter() - started) * 1000, 1)
    return {
        "ok": True,
        "entrypoint": entry,
        "runs": len(points),
        "total_ms": total_ms,
        "points": points,
        "fit": _fit_growth(points),
    }


def _fit_growth(points: list) -> str:
    """Very rough growth label from the first vs last timed point."""
    usable = [p for p in points if p["ms"] > 0]
    if len(usable) < 4:
        return "insufficient data"
    a, b = usable[0], usable[-1]
    n_ratio = b["n"] / a["n"] if a["n"] else 1
    t_ratio = b["ms"] / a["ms"] if a["ms"] else 1
    if n_ratio <= 1:
        return "flat"
    import math
    exp = math.log(t_ratio) / math.log(n_ratio)
    if exp < 0.4:
        return "≈ O(1)"
    if exp < 1.3:
        return "≈ O(n)"
    if exp < 1.7:
        return "≈ O(n log n)"
    if exp < 2.4:
        return "≈ O(n²)"
    if exp < 3.4:
        return "≈ O(n³)"
    return f"≈ O(n^{exp:.1f})"


class handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: dict) -> None:
        blob = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("content-length", 0) or 0)
            body = self.rfile.read(length) if length else b""
            self._send(200, _run(body))
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"ok": False, "error": str(exc)})
