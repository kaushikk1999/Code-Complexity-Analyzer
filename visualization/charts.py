"""Plotly chart builders."""

from __future__ import annotations

from typing import Dict, List

import plotly.graph_objects as go

from benchmarking.metrics import BenchmarkResult, ScalingBenchmarkResult
from scoring.optimizer_score import ScoreBreakdown

PLOT_TEMPLATE = "plotly_dark"


def _layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.26)",
        margin=dict(l=35, r=20, t=52, b=35),
        font=dict(color="#dbeafe"),
        title_font=dict(size=18, color="#f8fafc"),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.14)", zerolinecolor="rgba(148,163,184,0.18)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.14)", zerolinecolor="rgba(148,163,184,0.18)")
    return fig


def runtime_chart(benchmark: BenchmarkResult) -> go.Figure:
    x = [run.run_index for run in benchmark.runs]
    y = [run.runtime_ms for run in benchmark.runs]
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                line=dict(color="#22d3ee", width=3),
                marker=dict(size=8, color="#67e8f9"),
                fill="tozeroy",
                fillcolor="rgba(34,211,238,0.12)",
            )
        ]
    )
    fig.update_xaxes(title="Run")
    fig.update_yaxes(title="Runtime (ms)")
    return _layout(fig, "Runtime Per Run")


def memory_chart(benchmark: BenchmarkResult) -> go.Figure:
    x = [run.run_index for run in benchmark.runs]
    y = [run.peak_memory_kb for run in benchmark.runs]
    fig = go.Figure(
        data=[
            go.Bar(
                x=x,
                y=y,
                marker=dict(color="#a78bfa"),
                hovertemplate="Run %{x}<br>Peak %{y:.2f} KB<extra></extra>",
            )
        ]
    )
    fig.update_xaxes(title="Run")
    fig.update_yaxes(title="Peak Memory (KB)")
    return _layout(fig, "Peak Traced Memory")


def score_breakdown_chart(score: ScoreBreakdown) -> go.Figure:
    names = list(score.components.keys())
    values = list(score.components.values())
    fig = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=names,
                orientation="h",
                marker=dict(color=["#22d3ee", "#60a5fa", "#34d399", "#f59e0b", "#a78bfa", "#c084fc", "#fb7185"]),
            )
        ]
    )
    fig.update_xaxes(title="Points Earned")
    fig.update_yaxes(title="")
    return _layout(fig, "Optimization Score Components")


def history_chart(history: List[Dict[str, float]]) -> go.Figure:
    x = list(range(1, len(history) + 1))
    runtime = [item.get("peak_ms", item.get("avg_ms", 0.0)) for item in history]
    memory = [item.get("peak_kb", 0.0) for item in history]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=runtime,
            mode="lines+markers",
            name="Peak runtime (ms)",
            line=dict(color="#22d3ee", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=memory,
            mode="lines+markers",
            name="Peak memory (KB)",
            yaxis="y2",
            line=dict(color="#a78bfa", width=3),
        )
    )
    fig.update_layout(
        yaxis=dict(title="Runtime (ms)"),
        yaxis2=dict(title="Memory (KB)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return _layout(fig, "Benchmark History Trend")


def scaling_chart(scaling: ScalingBenchmarkResult) -> go.Figure:
    successful = [point for point in scaling.points if point.success]
    x = [point.input_size for point in successful]
    runtime = [point.max_ms for point in successful]
    memory = [point.max_peak_memory_kb for point in successful]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=runtime,
            mode="lines+markers",
            name="Peak runtime (ms)",
            line=dict(color="#34d399", width=3),
            marker=dict(size=8),
        )
    )
    fig.add_trace(
        go.Bar(
            x=x,
            y=memory,
            name="Peak memory (KB)",
            yaxis="y2",
            marker=dict(color="rgba(167,139,250,0.45)"),
        )
    )
    fig.update_layout(
        yaxis=dict(title="Runtime (ms)"),
        yaxis2=dict(title="Memory (KB)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title="Input size")
    return _layout(fig, f"Scaling Trend - Empirical {scaling.empirical_complexity}")


def score_gauge(score: ScoreBreakdown) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score.score,
            number={"suffix": "/100", "font": {"color": "#f8fafc"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                "bar": {"color": "#22d3ee"},
                "bgcolor": "rgba(15,23,42,0.55)",
                "borderwidth": 1,
                "bordercolor": "rgba(148,163,184,0.25)",
                "steps": [
                    {"range": [0, 38], "color": "rgba(251,113,133,0.35)"},
                    {"range": [38, 55], "color": "rgba(245,158,11,0.30)"},
                    {"range": [55, 72], "color": "rgba(96,165,250,0.25)"},
                    {"range": [72, 100], "color": "rgba(52,211,153,0.25)"},
                ],
            },
        )
    )
    return _layout(fig, "Optimization Quality Gauge")
