"""Vercel serverless function: LLM code generation via Ollama Cloud.

Powers two features:
  * mode="plan"     -> Algorithm Optimization Planner (natural-language problem -> plan)
  * mode="optimize" -> Code Analyzer AI rewrite (Python code -> optimized code)

Uses the Ollama Cloud chat API (https://ollama.com/api/chat) with a Bearer
key from the OLLAMA_API_KEY environment variable. No third-party deps.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler

OLLAMA_HOST = "https://ollama.com"
DEFAULT_MODEL = "gemma4:31b-cloud"
# Allowlisted Ollama Cloud models verified free on the free tier.
# (GLM / Kimi / Qwen / DeepSeek / MiniMax cloud tags require a paid plan.)
ALLOWED_MODELS = {
    "gemma4:31b-cloud",
    "gpt-oss:120b-cloud",
    "gpt-oss:20b-cloud",
    "gemini",  # Google Gemini API (needs GEMINI_API_KEY), not Ollama
}
# Google Gemini API: model id is configurable; GEMINI_API_KEY enables it.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
TIMEOUT = 55  # seconds; keep under the function maxDuration

PLAN_SYSTEM = (
    "You are a world-class competitive-programming coach. Given a problem "
    "statement, produce a concise optimization plan in GitHub-flavored "
    "Markdown with exactly these headings:\n"
    "## Problem Restatement\n## Brute-force Approach\n## Optimal Approach\n"
    "## Complexity\n## Reference Implementation (Python)\n## Edge Cases\n"
    "The Optimal Approach and Reference Implementation MUST achieve the lowest "
    "possible asymptotic time complexity for the problem, then the lowest "
    "auxiliary space (aim for O(1) extra space when feasible). Use the right "
    "data structure (hash map/set, heap, two pointers, sliding window, prefix "
    "sums, binary search, monotonic stack, DP with rolling arrays). State the "
    "exact Big-O for both brute force and optimal, and explain why the optimal "
    "cannot be beaten. Keep code in fenced ```python blocks. Do not invent "
    "constraints that were not given."
)

OPTIMIZE_SYSTEM = (
    "You are a world-class competitive-programming and Python performance "
    "expert. Given a Python snippet, return an optimized rewrite in "
    "GitHub-flavored Markdown with exactly these headings:\n"
    "## Summary\n## Optimized Code\n## Why It's Better\n"
    "## Complexity (before -> after)\n"
    "Achieve the LOWEST possible asymptotic time complexity first, then the "
    "lowest auxiliary space, preserving the public function names, arguments, "
    "and return contract. Replace nested scans and growing-list membership "
    "with hash maps/sets, heaps, two pointers, sliding windows, prefix sums, "
    "binary search, or O(1)-space DP as appropriate. Put the full rewrite in a "
    "single fenced ```python block, and state the exact before -> after "
    "time and space Big-O. If the code is already asymptotically optimal, say "
    "so and return only a minimal, correctness-preserving cleanup."
)


def _gemini_configured() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def active_models() -> list[str]:
    """Models usable right now (Gemini only when its key is set)."""
    return sorted(m for m in ALLOWED_MODELS if m != "gemini" or _gemini_configured())


def _gemini_chat(system: str, user: str, timeout: int = TIMEOUT) -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set on this deployment.")
    body = json.dumps({
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
    }).encode("utf-8")
    req = urllib.request.Request(
        GEMINI_URL.format(model=GEMINI_MODEL), data=body, method="POST",
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    parts = ((payload.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts).strip()


def _ollama_chat(api_key: str, model: str, system: str, user: str, timeout: int = TIMEOUT) -> str:
    """Chat with the selected model. 'gemini' routes to Google's API."""
    if model == "gemini":
        return _gemini_chat(system, user, timeout)
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return (payload.get("message", {}) or {}).get("content", "").strip()


def _run(raw_body: bytes) -> dict:
    try:
        data = json.loads(raw_body or b"{}")
    except (ValueError, TypeError):
        data = {}

    model = (data.get("model") or "").strip()
    if model not in ALLOWED_MODELS:
        model = DEFAULT_MODEL

    api_key = os.getenv("OLLAMA_API_KEY", "").strip()
    if model == "gemini" and not _gemini_configured():
        return {"ok": False, "error": "GEMINI_API_KEY is not set on this deployment. "
                "Add it in Vercel → Project → Settings → Environment Variables, then redeploy."}
    if model != "gemini" and not api_key:
        return {
            "ok": False,
            "error": "OLLAMA_API_KEY is not set on this deployment. Add it in "
            "Vercel → Project → Settings → Environment Variables, then redeploy.",
        }

    mode = (data.get("mode") or "plan").strip()
    if mode == "optimize":
        code = (data.get("code") or "").strip()
        if not code:
            return {"ok": False, "error": "No code provided."}
        system, user = OPTIMIZE_SYSTEM, f"Optimize this Python code:\n\n```python\n{code[:8000]}\n```"
    else:
        question = (data.get("question") or "").strip()
        if not question:
            return {"ok": False, "error": "No question provided."}
        system, user = PLAN_SYSTEM, f"Coding problem:\n\n{question[:8000]}"

    try:
        text = _ollama_chat(api_key, model, system, user)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")[:300] if exc.fp else ""
        src = "Gemini API" if model == "gemini" else "Ollama Cloud"
        return {"ok": False, "error": f"{src} error {exc.code}. {detail}"}
    except urllib.error.URLError as exc:
        return {"ok": False, "error": f"Could not reach the model API: {exc.reason}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Generation failed: {exc}"}

    if not text:
        return {"ok": False, "error": "Ollama returned an empty response."}
    return {"ok": True, "model": model, "text": text}


class handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

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

    def do_GET(self) -> None:
        configured = bool(os.getenv("OLLAMA_API_KEY", "").strip())
        self._send(200, {"ok": True, "default_model": DEFAULT_MODEL,
                         "models": active_models(), "key_configured": configured,
                         "gemini_configured": _gemini_configured(), "gemini_model": GEMINI_MODEL})
