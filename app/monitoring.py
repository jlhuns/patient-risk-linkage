"""
Every prediction gets logged to its own SQLite table (kind, input summary,
score, timestamp) — separate from the warehouse since this is operational
data, not clinical data. No Grafana/Prometheus, just enough to see volume
and score distribution over time.
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# Lambda's disk is read-only except /tmp, and /tmp resets on cold start and
# isn't shared across concurrent instances — known limitation, fine for a
# demo. Would move to DynamoDB for anything that needs to actually persist.
if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    DB_PATH = Path("/tmp/monitoring.db")
else:
    DB_PATH = Path(__file__).resolve().parent.parent / "monitoring.db"


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            input_summary TEXT,
            score REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    return conn


def log_prediction(kind: str, input_summary: str, score: float):
    conn = _connect()
    conn.execute(
        "INSERT INTO predictions (kind, input_summary, score, created_at) VALUES (?, ?, ?, ?)",
        (kind, input_summary, score, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def get_recent_predictions(limit: int = 25) -> list[dict]:
    conn = _connect()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT kind, input_summary, score, created_at FROM predictions ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_summary_stats() -> dict:
    conn = _connect()
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT kind, COUNT(*) AS n, AVG(score) AS avg_score, MIN(score) AS min_score, MAX(score) AS max_score
        FROM predictions GROUP BY kind
    """).fetchall()
    conn.close()
    return {r["kind"]: dict(r) for r in rows}
