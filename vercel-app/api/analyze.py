"""Vercel Python serverless function: static complexity analysis.

Wraps the AST-based `analyze_code` (pure stdlib) so the frontend can POST
Python source and get back the StaticAnalysisResult as JSON.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler

# The vendored `analyzer` package sits next to this file (see vercel.json
# includeFiles). Make it importable regardless of the function's cwd.
sys.path.insert(0, os.path.dirname(__file__))

from analyzer import analyze_code  # noqa: E402


def _analyze(raw_body: bytes) -> dict:
    try:
        payload = json.loads(raw_body or b"{}")
    except (ValueError, TypeError):
        payload = {}
    code = payload.get("code", "") if isinstance(payload, dict) else ""
    return analyze_code(code).to_dict()


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

    def do_OPTIONS(self) -> None:  # CORS preflight
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("content-length", 0) or 0)
            body = self.rfile.read(length) if length else b""
            self._send(200, _analyze(body))
        except Exception as exc:  # never leak a 500 without a message
            self._send(500, {"valid": False, "parse_error": str(exc)})

    def do_GET(self) -> None:
        self._send(200, {"ok": True, "usage": "POST {\"code\": \"...\"}"})
