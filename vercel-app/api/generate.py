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
# Allowlisted free Ollama Cloud models the UI can pick from.
ALLOWED_MODELS = {
    "glm-5.2:cloud",
    "kimi-k3:cloud",
    "gemma4:31b-cloud",
    "qwen3.5:cloud",
    "glm-5.1:cloud",
}
TIMEOUT = 55  # seconds; keep under the function maxDuration

PLAN_SYSTEM = (
    "You are an expert coding-interview algorithm coach. Given a problem "
    "statement, produce a concise, well-structured optimization plan in "
    "GitHub-flavored Markdown with exactly these headings:\n"
    "## Problem Restatement\n## Brute-force Approach\n## Optimal Approach\n"
    "## Complexity\n## Reference Implementation (Python)\n## Edge Cases\n"
    "Keep code inside fenced ```python blocks. Be precise about time/space "
    "complexity. Do not invent constraints that were not given."
)

OPTIMIZE_SYSTEM = (
    "You are an expert Python performance engineer. Given a Python snippet, "
    "return an optimized rewrite in GitHub-flavored Markdown with exactly "
    "these headings:\n## Summary\n## Optimized Code\n## Why It's Better\n"
    "## Complexity (before -> after)\n"
    "Preserve the public function names, arguments, and return contract. Put "
    "the full rewrite in a single fenced ```python block. If the code is "
    "already optimal, say so and return a minimal cleanup only."
)


def _ollama_chat(api_key: str, model: str, system: str, user: str) -> str:
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
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return (payload.get("message", {}) or {}).get("content", "").strip()


def _run(raw_body: bytes) -> dict:
    api_key = os.getenv("OLLAMA_API_KEY", "").strip()
    if not api_key:
        return {
            "ok": False,
            "error": "OLLAMA_API_KEY is not set on this deployment. Add it in "
            "Vercel → Project → Settings → Environment Variables, then redeploy.",
        }
    try:
        data = json.loads(raw_body or b"{}")
    except (ValueError, TypeError):
        data = {}

    model = (data.get("model") or "").strip()
    if model not in ALLOWED_MODELS:
        model = DEFAULT_MODEL

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
        return {"ok": False, "error": f"Ollama Cloud error {exc.code}. {detail}"}
    except urllib.error.URLError as exc:
        return {"ok": False, "error": f"Could not reach Ollama Cloud: {exc.reason}"}
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
                         "models": sorted(ALLOWED_MODELS), "key_configured": configured})
