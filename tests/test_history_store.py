from analyzer import analyze_code
from optimization import build_optimization_plan
from scoring import calculate_optimization_score
from utils import history_store


def test_history_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(history_store, "DB_PATH", tmp_path / "history.sqlite3")
    analysis = analyze_code("def total(values):\n    return sum(values)\n")
    score = calculate_optimization_score(analysis)
    plan = build_optimization_plan(analysis, score)
    record_id = history_store.save_analysis_record(
        session_name="Arrays",
        code=analysis.raw_code,
        entrypoint="total",
        benchmark_input='{"args": [[1, 2, 3]]}',
        analysis=analysis,
        score=score,
        plan=plan,
        report_markdown="# Report",
    )
    assert record_id == 1
    records = history_store.load_recent_records(session_name="Arrays")
    assert records[0]["entrypoint"] == "total"
    summary = history_store.progress_summary("Arrays")
    assert summary["count"] == 1
