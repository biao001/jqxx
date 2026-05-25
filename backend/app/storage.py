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


def get_stats(db_path: Path) -> dict[str, Any]:
    """跨会话聚合统计：总览、按天评分趋势、违规事件频次。"""
    rows = list_sessions(db_path, limit=1000)
    empty = {"total_sessions": 0, "total_duration": 0, "avg_score": 0, "total_events": 0,
             "by_day": [], "event_freq": [], "recent_scores": []}
    if not rows:
        return empty
    total_dur = sum(r["duration"] for r in rows)
    avg_score = round(sum(r["score_avg"] for r in rows) / len(rows))
    total_events = sum(r["events_count"] for r in rows)

    # 按天平均评分
    by_day: dict[str, list[float]] = {}
    for r in rows:
        day = time.strftime("%m-%d", time.localtime(r["created_at"]))
        by_day.setdefault(day, []).append(r["score_avg"])
    day_series = [{"day": d, "score": round(sum(v) / len(v))} for d, v in sorted(by_day.items())][-14:]

    # 违规事件频次(解析各会话 payload 的 events)
    freq: dict[str, int] = {}
    for r in rows[:300]:
        full = get_session(db_path, r["id"]) or {}
        for ev in full.get("events", []) or []:
            label = ev.get("label", "未知")
            freq[label] = freq.get(label, 0) + 1
    event_freq = sorted(({"label": k, "count": v} for k, v in freq.items()), key=lambda x: -x["count"])[:8]

    recent_scores = [round(r["score_avg"]) for r in rows[:20]][::-1]
    return {
        "total_sessions": len(rows),
        "total_duration": round(total_dur),
        "avg_score": avg_score,
        "total_events": total_events,
        "by_day": day_series,
        "event_freq": event_freq,
        "recent_scores": recent_scores,
    }


def delete_session(db_path: Path, sid: str) -> bool:
    if not db_path.exists():
        return False
    with _connect(db_path) as c:
        cur = c.execute("DELETE FROM sessions WHERE id = ?", (sid,))
    return cur.rowcount > 0
