"""HTML and Streamlit UI helper functions."""

from __future__ import annotations

import html
import json

import streamlit as st
import streamlit.components.v1 as components


def metric_card(label: str, value: str, caption: str = "") -> str:
    return f"""
<div class="metric-card">
  <div class="metric-label">{html.escape(label)}</div>
  <div class="metric-value">{html.escape(str(value))}</div>
  <div class="metric-caption">{html.escape(caption)}</div>
</div>
"""


def complexity_badge(label: str) -> str:
    return f'<span class="complexity-badge">{html.escape(label)}</span>'


def severity_pill(label: str) -> str:
    clean = html.escape(str(label).title())
    css = html.escape(str(label).lower().replace(" ", "-"))
    return f'<span class="severity-pill severity-{css}">{clean}</span>'


def glass_card(title: str, body: str) -> str:
    return f"""
<div class="glass-card">
  <div class="section-title">{html.escape(title)}</div>
  <div class="muted">{body}</div>
</div>
"""


def copyable_block(title: str, text: str, key: str) -> None:
    payload = json.dumps(text)
    safe_title = html.escape(title)
    components.html(
        f"""
<div style="font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: rgba(15,23,42,.78); border: 1px solid rgba(148,163,184,.24);
            border-radius: 16px; padding: 14px; color: #e5e7eb;">
  <div style="display:flex; justify-content:space-between; gap:12px; align-items:center; margin-bottom:10px;">
    <strong>{safe_title}</strong>
    <button id="copy-{key}"
      style="border:1px solid rgba(125,211,252,.35); background:rgba(34,211,238,.12);
             color:#dbeafe; border-radius:10px; padding:7px 10px; font-weight:700; cursor:pointer;">
      Copy
    </button>
  </div>
  <pre style="white-space:pre-wrap; margin:0; color:#cbd5e1; font-size:13px; line-height:1.45;">{html.escape(text)}</pre>
</div>
<script>
const text{key} = {payload};
document.getElementById("copy-{key}").onclick = async () => {{
  await navigator.clipboard.writeText(text{key});
  document.getElementById("copy-{key}").innerText = "Copied";
  setTimeout(() => document.getElementById("copy-{key}").innerText = "Copy", 1200);
}};
</script>
""",
        height=220,
    )


def render_empty_state(message: str) -> None:
    st.markdown(
        glass_card(
            "Waiting for analysis",
            html.escape(message),
        ),
        unsafe_allow_html=True,
    )
