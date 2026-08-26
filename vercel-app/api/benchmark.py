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
import copy
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
    """Return the entrypoint name. Top-level functions win; otherwise fall back
    to the first public method of the first class, as 'ClassName.method'
    (covers LeetCode-style `class Solution: def method(self, ...)`)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and not item.name.startswith("__"):
                    return f"{node.name}.{item.name}"
    return None


def _find_node_class(ns: dict):
    """Find a linked-list node class defined in the snippet (has a `.next`)."""
    for value in ns.values():
        if not isinstance(value, type):
            continue
        inst = None
        for attempt in ((0,), ()):
            try:
                inst = value(*attempt)
                break
            except Exception:
                continue
        if inst is not None and hasattr(inst, "next"):
            return value
    return None


def _make_linked_list(node_cls, size: int):
    """Build a linked list of `size` nodes from a node class like ListNode."""
    head = None
    for _ in range(max(1, size)):
        try:
            node = node_cls(random.randint(0, 9))
        except Exception:
            node = node_cls()
            setattr(node, "val", random.randint(0, 9))
        node.next = head
        head = node
    return head


def _resolve_callable(ns: dict, entry: str):
    """Resolve an entrypoint name from an exec'd namespace to a callable,
    instantiating the class for 'ClassName.method' entrypoints."""
    if "." in entry:
        cls_name, method = entry.split(".", 1)
        cls = ns.get(cls_name)
        if cls is None:
            return None
        try:
            instance = cls()
        except Exception:
            return None
        return getattr(instance, method, None)
    return ns.get(entry)


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


def _positional_params(func) -> list[str]:
    sig = inspect.signature(func)
    names = []
    for i, (pname, p) in enumerate(sig.parameters.items()):
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is not inspect.Parameter.empty and i > 1:
            continue  # let optional trailing args default
        names.append(pname or f"p{i}")
    return names


def _int_arg(size: int, first: bool) -> int:
    # small for the leading "n"-style arg (avoid exponential blow-ups), else size//2
    return max(1, min(size, 25)) if first else max(1, size // 2)


def _strategy_args(names: list[str], size: int, mode: str, node_cls=None) -> list:
    """Build a full positional-arg tuple under a given input-shape strategy."""
    args = []
    for i, nm in enumerate(names):
        if mode == "linked_list":
            args.append(_make_linked_list(node_cls, size))
        elif mode == "heuristic":
            args.append(_make_arg(nm, size))
        elif mode == "list_then_int":
            args.append([random.randint(0, size) for _ in range(size)] if i == 0 else _int_arg(size, False))
        elif mode == "all_list":
            args.append([random.randint(0, size) for _ in range(size)])
        elif mode == "pairs_then_int":
            args.append([[random.randint(0, size), random.randint(0, size)] for _ in range(size)]
                        if i == 0 else _int_arg(size, False))
        elif mode == "all_int":
            args.append(_int_arg(size, i == 0))
        else:
            args.append(_make_arg(nm, size))
    return args


# Ordered fallbacks: try each shape on a tiny probe, keep the first that runs.
_STRATEGIES = ("heuristic", "list_then_int", "all_list", "pairs_then_int", "all_int")


def _choose_strategy(func, names: list[str], node_cls=None) -> str | None:
    if not names:
        # zero-arg or fully-defaulted function: verify a bare call works.
        try:
            func()
            return "heuristic"
        except Exception:
            return None
    # Try a linked-list shape first when the snippet defines a node class.
    modes = (["linked_list"] + list(_STRATEGIES)) if node_cls is not None else list(_STRATEGIES)
    for mode in modes:
        try:
            func(*copy.deepcopy(_strategy_args(names, 6, mode, node_cls)))
            return mode
        except Exception:
            continue
    return None


def _build_args(func, size: int, node_cls=None):
    """Back-compat helper (used by the correctness check): pick a working shape."""
    names = _positional_params(func)
    mode = _choose_strategy(func, names, node_cls) or "heuristic"
    return _strategy_args(names, size, mode, node_cls)


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
        return {"ok": False, "error": "No function found to benchmark. Add a "
                "top-level `def` (or a class with a method) that takes a "
                "list/number argument."}

    ns: dict = {}
    try:
        exec(compile(code, "<snippet>", "exec"), ns)  # noqa: S102 (intentional)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Code failed to load: {exc}"}
    func = _resolve_callable(ns, entry)
    if not callable(func):
        return {"ok": False, "error": f"'{entry}' is not callable."}

    # Probe input shapes and keep the first that actually runs.
    node_cls = _find_node_class(ns)
    names = _positional_params(func)
    strategy = _choose_strategy(func, names, node_cls)
    if strategy is None:
        return {"ok": False, "error": f"Couldn't auto-generate inputs for '{entry}' "
                f"({len(names)} arg(s): {', '.join(names) or 'none'}). Its signature isn't "
                "auto-benchmarkable — try a function that takes a list/number."}

    # 30 increasing input sizes.
    sizes = [ (i + 1) * 20 for i in range(N_POINTS) ]
    points = []
    started = time.perf_counter()
    for size in sizes:
        try:
            args = _strategy_args(names, size, strategy, node_cls)
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
