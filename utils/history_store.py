"""SQLite-backed local history for practice sessions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from analyzer.models import StaticAnalysisResult
from benchmarking.metrics import BenchmarkResult, ScalingBenchmarkResult
from optimization.planner import OptimizationPlan
from scoring.optimizer_score import ScoreBreakdown

DB_PATH = Path("data/complexity_lab.sqlite3")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            session_name TEXT NOT NULL,
            code TEXT NOT NULL,
            entrypoint TEXT,
            benchmark_input TEXT,
            estimated_time TEXT,
            estimated_space TEXT,
            score INTEGER,
            severity TEXT,
            analysis_json TEXT,
            benchmark_json TEXT,
            scaling_json TEXT,
            plan_json TEXT,
            report_markdown TEXT
        )
        """
    )
    connection.commit()
    return connection


def save_analysis_record(
    session_name: str,
    code: str,
    entrypoint: str,
    benchmark_input: str,
    analysis: StaticAnalysisResult,
    score: ScoreBreakdown,
    plan: OptimizationPlan,
    report_markdown: str,
    benchmark: Optional[BenchmarkResult] = None,
    scaling: Optional[ScalingBenchmarkResult] = None,
) -> int:
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO analysis_history (
                session_name, code, entrypoint, benchmark_input, estimated_time, estimated_space,
                score, severity, analysis_json, benchmark_json, scaling_json, plan_json, report_markdown
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_name or "General",
                code,
                entrypoint,
                benchmark_input,
                analysis.estimated_time,
                analysis.estimated_space,
                score.score,
                score.severity,
                json.dumps(analysis.to_dict()),
                json.dumps(benchmark.to_dict()) if benchmark else None,
                json.dumps(scaling.to_dict()) if scaling else None,
                json.dumps(plan.to_dict()),
                report_markdown,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def load_recent_records(limit: int = 20, session_name: str = "") -> List[Dict[str, Any]]:
    query = """
        SELECT id, created_at, session_name, entrypoint, estimated_time, estimated_space, score, severity
        FROM analysis_history
    """
    params: List[Any] = []
    if session_name:
        query += " WHERE session_name = ?"
        params.append(session_name)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with _connect() as connection:
        rows = connection.execute(query, params).fetchall()
    return [
        {
            "id": row[0],
            "created_at": row[1],
            "session": row[2],
            "entrypoint": row[3],
            "time": row[4],
            "space": row[5],
            "score": row[6],
            "severity": row[7],
        }
        for row in rows
    ]


def progress_summary(session_name: str = "") -> Dict[str, Any]:
    records = load_recent_records(limit=200, session_name=session_name)
    if not records:
        return {
            "count": 0,
            "average_score": 0,
            "best_score": 0,
            "latest_score": 0,
            "weakest_area": "No saved analyses yet",
        }
    scores = [int(record["score"] or 0) for record in records]
    severities: Dict[str, int] = {}
    for record in records:
        severities[record["severity"]] = severities.get(record["severity"], 0) + 1
    weakest = max(severities, key=severities.get)
    return {
        "count": len(records),
        "average_score": round(sum(scores) / len(scores), 1),
        "best_score": max(scores),
        "latest_score": scores[0],
        "weakest_area": weakest,
    }
