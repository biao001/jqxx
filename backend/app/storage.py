"""会话持久化：用 SQLite 存每次行程的汇总 + 事件 + 评分曲线，支持历史回看。"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at REAL,
                source TEXT,
                duration REAL,
                score_avg REAL,
                score_min REAL,
                behavior_avg REAL,
                fatigue_avg REAL,
                events_count INTEGER,
                payload TEXT
            )
            """
        )


def create_session(db_path: Path, data: dict[str, Any]) -> str:
    sid = uuid.uuid4().hex[:12]
    with _connect(db_path) as c:
        c.execute(
            "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                sid,
                time.time(),
                str(data.get("source", ""))[:120],
                float(data.get("duration", 0) or 0),
                float(data.get("score_avg", 0) or 0),
                float(data.get("score_min", 0) or 0),
                float(data.get("behavior_avg", 0) or 0),
                float(data.get("fatigue_avg", 0) or 0),
                int(data.get("events_count", 0) or 0),
                json.dumps(data, ensure_ascii=False),
            ),
        )
    return sid


def list_sessions(db_path: Path, limit: int = 50) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    with _connect(db_path) as c:
        rows = c.execute(
            "SELECT id, created_at, source, duration, score_avg, score_min, "
            "behavior_avg, fatigue_avg, events_count FROM sessions "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(db_path: Path, sid: str) -> dict[str, Any] | None:
    if not db_path.exists():
        return None
    with _connect(db_path) as c:
        row = c.execute("SELECT payload FROM sessions WHERE id = ?", (sid,)).fetchone()
    return json.loads(row["payload"]) if row else None


def delete_session(db_path: Path, sid: str) -> bool:
    if not db_path.exists():
        return False
    with _connect(db_path) as c:
        cur = c.execute("DELETE FROM sessions WHERE id = ?", (sid,))
    return cur.rowcount > 0
