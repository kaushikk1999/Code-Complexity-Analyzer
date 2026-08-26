"""Multi-LLM optimize-and-rank pipeline.

Workflow (per the product spec):
  1. Benchmark the original code -> baseline (time, memory, cpu, complexity,
     LOC, cyclomatic complexity, readability).
  2. Ask ALL configured free models IN PARALLEL for an optimized rewrite.
  3. Validate each candidate: re-benchmark + correctness vs the original.
  4. Weighted ranking -> return the Top 3 with full metrics.

PLATFORM NOTE: executes code, so it is a local/best-effort feature (not a
hardened sandbox). "Configured models" are the free Ollama Cloud models in
generate.ALLOWED_MODELS; Claude/GPT/Gemini would need their own keys.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler

_HERE = os.path.dirname(__file__)
import sys
sys.path.insert(0, _HERE)
from analyzer import analyze_code  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load("generate")
bench = _load("benchmark")

# Ranking weights (sum of non-correctness terms = 0.60; correctness = 0.40).
W_CORRECT, W_TIME, W_MEM, W_CX, W_READ = 0.40, 0.20, 0.15, 0.15, 0.10
CX_RANK = {"O(1)": 0, "O(log n)": 1, "O(n)": 2, "O(n log n)": 3,
           "O(n^2)": 4, "O(n^2 log n)": 5, "O(n^3)": 6, "O(2^n)": 8, "Unknown": 4}

OPT_SYSTEM = (
    "You are an expert Python performance engineer. Rewrite the given function "
    "to be faster and lighter while preserving identical behavior, argument "
    "names, and return contract. Prefer better algorithms and data structures "
    "over micro-optimizations. Keep it simple, readable, and beginner-friendly. "
    "Return ONLY a single fenced ```python code block containing the full "
    "rewrite, then 2-4 sentences explaining the key optimizations."
)


# ---------- static metrics ----------
def _loc(code: str) -> int:
    return sum(1 for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("#"))


def _cyclomatic(code: str) -> int:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    count = 1
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler,
                             ast.With, ast.Assert, ast.comprehension)):
            count += 1
        elif isinstance(node, ast.BoolOp):
            count += len(node.values) - 1
        elif isinstance(node, ast.IfExp):
            count += 1
    return count


def _max_nesting(code: str) -> int:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    best = 0

    def walk(node, depth):
        nonlocal best
        best = max(best, depth)
        for child in ast.iter_child_nodes(node):
            inc = 1 if isinstance(child, (ast.For, ast.While, ast.If, ast.With, ast.Try)) else 0
            walk(child, depth + inc)

    walk(tree, 0)
    return best


def _readability(code: str) -> int:
    """0-100 heuristic: penalize length, nesting, cyclomatic, long lines."""
    loc = _loc(code) or 1
    nest = _max_nesting(code)
    cx = _cyclomatic(code)
    long_lines = sum(1 for ln in code.splitlines() if len(ln) > 90)
    score = 100
    score -= max(0, nest - 2) * 12
    score -= max(0, cx - 5) * 4
    score -= max(0, loc - 25) * 1.5
    score -= long_lines * 5
    return int(max(5, min(100, score)))


def _static_metrics(code: str) -> dict:
    a = analyze_code(code)
    return {
        "time_complexity": a.estimated_time if a.valid else "Unknown",
        "space_complexity": a.estimated_space if a.valid else "Unknown",
        "loc": _loc(code),
        "cyclomatic": _cyclomatic(code),
        "readability": _readability(code),
        "maintainability": int(round(0.6 * _readability(code)
                                     + 0.4 * max(0, 100 - _cyclomatic(code) * 5))),
    }


# ---------- dynamic metrics ----------
def _benchmark(code: str, entrypoint: str = "") -> dict:
    """Return {ok, time_ms, peak_kb, cpu_ms, total_ms, fit, entrypoint}.

    Passing the known entrypoint lets candidates that define it via assignment
    (e.g. a lambda) or keep the same name still benchmark."""
    payload = {"code": code}
    if entrypoint:
        payload["entrypoint"] = entrypoint
    res = bench._run(json.dumps(payload).encode())
    if not res.get("ok") and entrypoint:
        # entrypoint may have been renamed — retry with auto-detection.
        res = bench._run(json.dumps({"code": code}).encode())
    if not res.get("ok"):
        return {"ok": False, "error": res.get("error", "benchmark failed")}
    pts = res["points"]
    tail = pts[-5:] if len(pts) >= 5 else pts
    time_ms = round(sum(p["ms"] for p in tail) / len(tail), 4)
    cpu_ms = round(sum(p.get("cpu_ms", 0.0) for p in tail) / len(tail), 4)
    peak_kb = round(max(p["kb"] for p in pts), 1)
    return {"ok": True, "time_ms": time_ms, "cpu_ms": cpu_ms, "peak_kb": peak_kb,
            "total_ms": res["total_ms"], "fit": res["fit"], "entrypoint": res["entrypoint"]}


def _load_func(code: str, entry: str):
    ns: dict = {}
    exec(compile(code, "<cand>", "exec"), ns)  # noqa: S102
    return bench._resolve_callable(ns, entry)


def _correctness(original: str, candidate: str, entry: str) -> float:
    """Fraction of shared inputs where candidate output == original output."""
    try:
        of, cf = _load_func(original, entry), _load_func(candidate, entry)
    except Exception:
        return 0.0
    if not callable(of) or not callable(cf):
        return 0.0
    total, ok = 0, 0
    for size in (15, 40, 90):
        try:
            args = bench._build_args(of, size)
        except Exception:
            continue
        total += 1
        try:
            a = of(*copy.deepcopy(args))
            b = cf(*copy.deepcopy(args))
            if repr(a) == repr(b):
                ok += 1
        except Exception:
            pass
    return ok / total if total else 0.0


# ---------- LLM generation (parallel) ----------
GEN_TIMEOUT = 150  # per-model read timeout; big models are slow
GEN_RETRIES = 2    # extra attempts on timeout/transient errors


def _looks_like_code(text: str) -> bool:
    return bool(re.search(r"^\s*(def|class|from|import)\s", text, re.M))


def _strip_thinking(text: str) -> str:
    # Some models emit <think>...</think> or "Thinking..." preambles.
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.I)
    return text.strip()


def _extract_code(text: str) -> tuple[str, str]:
    """Lenient extraction: ```python fence -> any ``` fence -> raw code body."""
    text = _strip_thinking(text)
    # 1) a python-tagged fence
    m = re.search(r"```(?:python|py)\s*([\s\S]*?)```", text, re.I)
    # 2) any fenced block whose contents look like code
    if not m:
        for blk in re.finditer(r"```[a-zA-Z0-9]*\s*([\s\S]*?)```", text):
            if _looks_like_code(blk.group(1)):
                m = blk
                break
    if m:
        code = m.group(1).strip()
        explanation = re.sub(r"```[a-zA-Z0-9]*[\s\S]*?```", "", text).strip()
        return code, explanation[:600]
    # 3) no fences at all — if the whole reply is basically code, use it
    if _looks_like_code(text):
        return text.strip(), ""
    return "", text[:600]


def _generate_one(api_key: str, model: str, code: str) -> dict:
    user = ("Optimize this Python code. Return the full rewrite inside a single "
            f"```python code block.\n\n```python\n{code[:8000]}\n```")
    last_err = ""
    for attempt in range(1 + GEN_RETRIES):
        try:
            text = gen._ollama_chat(api_key, model, OPT_SYSTEM, user, timeout=GEN_TIMEOUT)
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)[:200]
            continue  # retry on timeout / transient network error
        ccode, expl = _extract_code(text)
        if ccode:
            return {"model": model, "ok": True, "code": ccode, "explanation": expl}
        last_err = "No code block returned."
        # retry once more asking more forcefully handled by same prompt
    return {"model": model, "ok": False, "error": last_err or "generation failed"}


# ---------- scoring ----------
def _score(baseline: dict, cand: dict) -> float:
    corr = cand["correctness"]
    b_ms = baseline["dynamic"]["time_ms"] or 1e-6
    c_ms = cand["dynamic"]["time_ms"] or 1e-6
    b_kb = baseline["dynamic"]["peak_kb"] or 1e-6
    c_kb = cand["dynamic"]["peak_kb"] or 1e-6
    time_score = min(b_ms / c_ms, 2.0) / 2.0
    mem_score = min(b_kb / c_kb, 2.0) / 2.0
    b_rank = CX_RANK.get(baseline["static"]["time_complexity"], 4)
    c_rank = CX_RANK.get(cand["static"]["time_complexity"], 4)
    cx_score = max(0.0, min(1.0, (b_rank - c_rank + 2) / 4.0))
    read_score = cand["static"]["readability"] / 100.0
    overall = (W_CORRECT * corr + W_TIME * time_score + W_MEM * mem_score
               + W_CX * cx_score + W_READ * read_score)
    return round(overall * 100, 1)


def _pct(base: float, cand: float) -> float:
    if not base:
        return 0.0
    return round((base - cand) / base * 100, 1)


def _run(raw_body: bytes) -> dict:
    api_key = os.getenv("OLLAMA_API_KEY", "").strip()
    if not api_key:
        return {"ok": False, "error": "OLLAMA_API_KEY is not set."}
    try:
        data = json.loads(raw_body or b"{}")
    except (ValueError, TypeError):
        data = {}
    code = (data.get("code") or "").strip()
    if not code:
        return {"ok": False, "error": "No code provided."}
    guard = bench._guard(code)
    if guard:
        return {"ok": False, "error": guard}
    entry = bench._first_function(code)
    if not entry:
        return {"ok": False, "error": "No top-level function found."}

    # 1. Baseline
    base_dyn = _benchmark(code)
    if not base_dyn.get("ok"):
        return {"ok": False, "error": f"Baseline benchmark failed: {base_dyn.get('error')}"}
    baseline = {"entrypoint": entry, "static": _static_metrics(code), "dynamic": base_dyn}

    # 2. Parallel generation
    models = sorted(gen.ALLOWED_MODELS)
    raw_candidates = []
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        futs = {pool.submit(_generate_one, api_key, m, code): m for m in models}
        for fut in as_completed(futs):
            raw_candidates.append(fut.result())

    # 3. Validate + benchmark each candidate
    evaluated, failures = [], []
    for c in raw_candidates:
        if not c.get("ok"):
            failures.append({"model": c["model"], "error": c.get("error", "generation failed")})
            continue
        dyn = _benchmark(c["code"], entry)
        if not dyn.get("ok"):
            failures.append({"model": c["model"], "error": f"candidate benchmark failed: {dyn.get('error')}"})
            continue
        corr = _correctness(code, c["code"], entry)
        cand = {
            "model": c["model"], "code": c["code"], "explanation": c["explanation"],
            "static": _static_metrics(c["code"]), "dynamic": dyn, "correctness": corr,
        }
        cand["score"] = _score(baseline, cand)
        cand["time_improvement_pct"] = _pct(base_dyn["time_ms"], dyn["time_ms"])
        cand["mem_improvement_pct"] = _pct(base_dyn["peak_kb"], dyn["peak_kb"])
        evaluated.append(cand)

    # 4. Rank: correct solutions first, then by score.
    evaluated.sort(key=lambda x: (x["correctness"] >= 0.999, x["score"]), reverse=True)
    return {
        "ok": True,
        "baseline": baseline,
        "models_tried": models,
        "top": evaluated[:3],
        "all_count": len(evaluated),
        "failures": failures,
    }


class handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: dict) -> None:
        blob = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
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
