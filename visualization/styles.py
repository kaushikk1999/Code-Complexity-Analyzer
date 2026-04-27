"""Custom CSS for the Streamlit dashboard."""

from __future__ import annotations

import streamlit as st


def inject_global_styles() -> None:
    st.markdown(
        """
<style>
:root {
  --bg: #0b1020;
  --panel: rgba(15, 23, 42, 0.74);
  --panel-strong: rgba(17, 24, 39, 0.92);
  --text: #e5e7eb;
  --muted: #94a3b8;
  --line: rgba(148, 163, 184, 0.22);
  --cyan: #22d3ee;
  --blue: #60a5fa;
  --green: #34d399;
  --amber: #f59e0b;
  --red: #fb7185;
  --violet: #a78bfa;
}

.stApp {
  background:
    radial-gradient(circle at 12% 8%, rgba(34, 211, 238, 0.16), transparent 24rem),
    radial-gradient(circle at 88% 12%, rgba(167, 139, 250, 0.14), transparent 22rem),
    linear-gradient(135deg, #080b16 0%, #0f172a 52%, #111827 100%);
  color: var(--text);
}

section[data-testid="stSidebar"] {
  background: rgba(8, 13, 28, 0.90);
  border-right: 1px solid var(--line);
}

.block-container {
  padding-top: 2rem;
  padding-bottom: 3rem;
  max-width: 1280px;
}

.hero {
  border: 1px solid rgba(125, 211, 252, 0.22);
  background:
    linear-gradient(135deg, rgba(14, 165, 233, 0.28), rgba(124, 58, 237, 0.18)),
    rgba(15, 23, 42, 0.72);
  box-shadow: 0 26px 70px rgba(2, 6, 23, 0.42);
  border-radius: 24px;
  padding: 30px 32px;
  margin-bottom: 22px;
}

.hero h1 {
  font-size: clamp(2.1rem, 4vw, 4.2rem);
  line-height: 1;
  margin: 0 0 12px 0;
  letter-spacing: 0;
  color: #f8fafc;
}

.hero p {
  color: #cbd5e1;
  max-width: 760px;
  font-size: 1.05rem;
  margin: 0;
}

.hero-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 18px;
}

.pill, .severity-pill, .complexity-badge {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 5px 10px;
  font-size: 0.78rem;
  font-weight: 700;
  border: 1px solid rgba(148, 163, 184, 0.24);
  background: rgba(15, 23, 42, 0.62);
  color: #dbeafe;
}

.metric-card {
  min-height: 128px;
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 18px;
  background: linear-gradient(180deg, rgba(30, 41, 59, 0.82), rgba(15, 23, 42, 0.72));
  box-shadow: 0 18px 48px rgba(2, 6, 23, 0.28);
}

.metric-label {
  color: var(--muted);
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-weight: 800;
}

.metric-value {
  color: #f8fafc;
  font-size: 1.85rem;
  line-height: 1.1;
  font-weight: 850;
  margin-top: 10px;
}

.metric-caption {
  color: #a7b5c7;
  font-size: 0.86rem;
  margin-top: 8px;
}

.glass-card {
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 18px;
  background: rgba(15, 23, 42, 0.68);
  box-shadow: 0 16px 42px rgba(2, 6, 23, 0.22);
  margin-bottom: 14px;
}

.section-title {
  font-size: 1.14rem;
  color: #f8fafc;
  font-weight: 820;
  margin: 4px 0 10px 0;
}

.muted {
  color: var(--muted);
}

.complexity-badge {
  color: #cffafe;
  background: linear-gradient(135deg, rgba(34, 211, 238, 0.18), rgba(59, 130, 246, 0.18));
}

.severity-excellent, .severity-strong {
  color: #d1fae5;
  background: rgba(16, 185, 129, 0.16);
}
.severity-moderate {
  color: #fef3c7;
  background: rgba(245, 158, 11, 0.16);
}
.severity-risky, .severity-critical, .severity-high {
  color: #ffe4e6;
  background: rgba(251, 113, 133, 0.16);
}
.severity-medium {
  color: #fef3c7;
  background: rgba(245, 158, 11, 0.16);
}
.severity-low {
  color: #dbeafe;
  background: rgba(96, 165, 250, 0.16);
}

div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
  border-radius: 14px;
  border: 1px solid rgba(148, 163, 184, 0.26);
  background: rgba(2, 6, 23, 0.34);
  color: #e5e7eb;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
  background: rgba(15, 23, 42, 0.48);
  border: 1px solid var(--line);
  padding: 8px;
  border-radius: 16px;
}

.stTabs [data-baseweb="tab"] {
  border-radius: 12px;
  color: #cbd5e1;
  padding: 8px 14px;
}

.stTabs [aria-selected="true"] {
  background: rgba(34, 211, 238, 0.14);
  color: #f8fafc;
}

button[kind="primary"], div[data-testid="stButton"] button {
  border-radius: 12px;
  border: 1px solid rgba(125, 211, 252, 0.30);
  font-weight: 800;
}

pre, code {
  border-radius: 12px;
}

@media (max-width: 760px) {
  .hero {
    padding: 22px;
    border-radius: 18px;
  }
  .metric-card {
    min-height: auto;
  }
}
</style>
""",
        unsafe_allow_html=True,
    )
