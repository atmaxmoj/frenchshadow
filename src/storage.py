"""Persistent storage for shadow-reader practice attempts.

Each recorded sentence is saved as a row in SQLite; the audio blob is kept on
disk under recordings/ and referenced by path.  A `videos` table keeps the
latest metadata and last-practiced sentence for dashboard/continue flows.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).with_suffix("").parent.parent
DB_PATH = _PROJECT_ROOT / "data" / "attempts.db"
RECORDINGS_DIR = _PROJECT_ROOT / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Attempt:
    id: str
    video_id: str
    sentence_idx: int
    sentence_text: str
    language: str
    overall_score: float
    analysis: dict[str, Any]
    recording_path: str
    duration_s: float
    created_at: str


@dataclass(frozen=True)
class VideoProgress:
    video_id: str
    title: str
    thumbnail: str
    language: str
    total_sentences: int
    last_sentence_idx: int
    last_practiced_at: str | None
    attempt_count: int
    sentence_attempt_count: int


def _init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # Core tables
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS attempts (
            id TEXT PRIMARY KEY,
            video_id TEXT NOT NULL,
            sentence_idx INTEGER NOT NULL,
            sentence_text TEXT NOT NULL,
            language TEXT NOT NULL,
            overall_score REAL NOT NULL,
            analysis_json TEXT NOT NULL,
            recording_path TEXT NOT NULL,
            duration_s REAL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_attempts_video ON attempts(video_id);
        CREATE INDEX IF NOT EXISTS idx_attempts_sentence ON attempts(video_id, sentence_idx);
        CREATE INDEX IF NOT EXISTS idx_attempts_created ON attempts(created_at);

        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            thumbnail TEXT NOT NULL,
            language TEXT NOT NULL,
            total_sentences INTEGER NOT NULL DEFAULT 0,
            last_sentence_idx INTEGER NOT NULL DEFAULT 0,
            last_practiced_at TEXT
        );
        """
    )

    # Lightweight migrations for older DBs
    _add_column_if_missing(conn, "attempts", "duration_s", "REAL DEFAULT 0")
    _add_column_if_missing(conn, "videos", "total_sentences", "INTEGER NOT NULL DEFAULT 0")
    _add_column_if_missing(conn, "videos", "last_sentence_idx", "INTEGER NOT NULL DEFAULT 0")
    _add_column_if_missing(conn, "videos", "last_practiced_at", "TEXT")

    conn.commit()
    return conn


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    if not any(r["name"] == column for r in rows):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


_CONN: sqlite3.Connection | None = None


def _conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        _CONN = _init_db()
    return _CONN


def touch_video(
    video_id: str,
    title: str,
    thumbnail: str,
    language: str,
    total_sentences: int = 0,
    last_sentence_idx: int = 0,
) -> None:
    """Upsert video metadata used by the dashboard."""
    conn = _conn()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO videos (video_id, title, thumbnail, language, total_sentences,
                            last_sentence_idx, last_practiced_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(video_id) DO UPDATE SET
            title = excluded.title,
            thumbnail = excluded.thumbnail,
            language = excluded.language,
            total_sentences = excluded.total_sentences,
            last_sentence_idx = excluded.last_sentence_idx,
            last_practiced_at = excluded.last_practiced_at
        """,
        (video_id, title, thumbnail, language, total_sentences, last_sentence_idx, now),
    )
    conn.commit()


def save_attempt(
    video_id: str,
    sentence_idx: int,
    sentence_text: str,
    language: str,
    audio_bytes: bytes,
    analysis: dict[str, Any],
    duration_s: float = 0,
) -> Attempt:
    """Persist one practice attempt and return its record."""
    attempt_id = uuid.uuid4().hex
    recording_filename = f"{attempt_id}.webm"
    recording_path = RECORDINGS_DIR / recording_filename
    recording_path.write_bytes(audio_bytes)

    created_at = datetime.now(timezone.utc).isoformat()
    overall_score = float(analysis.get("overall_score", 0)) if analysis else 0.0

    conn = _conn()
    conn.execute(
        """
        INSERT INTO attempts (id, video_id, sentence_idx, sentence_text, language,
                              overall_score, analysis_json, recording_path, duration_s, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            attempt_id,
            video_id,
            sentence_idx,
            sentence_text,
            language,
            overall_score,
            json.dumps(analysis, ensure_ascii=False),
            str(recording_path),
            duration_s,
            created_at,
        ),
    )
    conn.execute(
        """
        UPDATE videos
        SET last_sentence_idx = ?, last_practiced_at = ?
        WHERE video_id = ?
        """,
        (sentence_idx, created_at, video_id),
    )
    conn.commit()

    return Attempt(
        id=attempt_id,
        video_id=video_id,
        sentence_idx=sentence_idx,
        sentence_text=sentence_text,
        language=language,
        overall_score=overall_score,
        analysis=analysis,
        recording_path=str(recording_path),
        duration_s=duration_s,
        created_at=created_at,
    )


def get_attempts(video_id: str, sentence_idx: int | None = None) -> list[Attempt]:
    """Return attempts for a video, optionally filtered by sentence index."""
    conn = _conn()
    if sentence_idx is None:
        rows = conn.execute(
            "SELECT * FROM attempts WHERE video_id = ? ORDER BY sentence_idx, created_at",
            (video_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM attempts WHERE video_id = ? AND sentence_idx = ? ORDER BY created_at",
            (video_id, sentence_idx),
        ).fetchall()
    return [_row_to_attempt(row) for row in rows]


def get_attempt(attempt_id: str) -> Attempt | None:
    """Return a single attempt by id, or None."""
    conn = _conn()
    row = conn.execute("SELECT * FROM attempts WHERE id = ?", (attempt_id,)).fetchone()
    return _row_to_attempt(row) if row else None


def _row_to_attempt(row: sqlite3.Row) -> Attempt:
    return Attempt(
        id=row["id"],
        video_id=row["video_id"],
        sentence_idx=row["sentence_idx"],
        sentence_text=row["sentence_text"],
        language=row["language"],
        overall_score=row["overall_score"],
        analysis=json.loads(row["analysis_json"]),
        recording_path=row["recording_path"],
        duration_s=row["duration_s"] or 0,
        created_at=row["created_at"],
    )


def get_recording_path(attempt_id: str) -> Path | None:
    """Return the filesystem path for an attempt's recording, if it exists."""
    attempt = get_attempt(attempt_id)
    if attempt is None:
        return None
    path = Path(attempt.recording_path)
    return path if path.exists() else None


def get_recent_videos(limit: int = 20) -> list[VideoProgress]:
    """Return videos ordered by most recent practice, with attempt counts."""
    conn = _conn()
    rows = conn.execute(
        """
        SELECT
            v.video_id,
            v.title,
            v.thumbnail,
            v.language,
            v.total_sentences,
            v.last_sentence_idx,
            v.last_practiced_at,
            COUNT(a.id) AS attempt_count,
            COUNT(DISTINCT a.sentence_idx) AS sentence_attempt_count
        FROM videos v
        LEFT JOIN attempts a ON a.video_id = v.video_id
        GROUP BY v.video_id
        ORDER BY v.last_practiced_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [_row_to_video(row) for row in rows]


def get_video_progress(video_id: str) -> VideoProgress | None:
    conn = _conn()
    row = conn.execute(
        """
        SELECT
            v.video_id,
            v.title,
            v.thumbnail,
            v.language,
            v.total_sentences,
            v.last_sentence_idx,
            v.last_practiced_at,
            COUNT(a.id) AS attempt_count,
            COUNT(DISTINCT a.sentence_idx) AS sentence_attempt_count
        FROM videos v
        LEFT JOIN attempts a ON a.video_id = v.video_id
        WHERE v.video_id = ?
        GROUP BY v.video_id
        """,
        (video_id,),
    ).fetchone()
    return _row_to_video(row) if row else None


def _row_to_video(row: sqlite3.Row) -> VideoProgress:
    return VideoProgress(
        video_id=row["video_id"],
        title=row["title"],
        thumbnail=row["thumbnail"],
        language=row["language"],
        total_sentences=row["total_sentences"],
        last_sentence_idx=row["last_sentence_idx"],
        last_practiced_at=row["last_practiced_at"],
        attempt_count=row["attempt_count"],
        sentence_attempt_count=row["sentence_attempt_count"],
    )


def get_stats() -> dict[str, Any]:
    """Return aggregate practice statistics."""
    conn = _conn()
    row = conn.execute(
        """
        SELECT
            COUNT(DISTINCT video_id) AS videos,
            COUNT(*) AS attempts,
            COUNT(DISTINCT sentence_idx || ':' || video_id) AS sentences,
            COALESCE(SUM(duration_s), 0) AS total_seconds,
            COUNT(DISTINCT DATE(created_at)) AS days
        FROM attempts
        """
    ).fetchone()

    # Streak: consecutive days with at least one attempt, ending today/yesterday.
    dates = [
        d[0]
        for d in conn.execute(
            "SELECT DISTINCT DATE(created_at) AS d FROM attempts ORDER BY d DESC"
        ).fetchall()
    ]
    streak = 0
    if dates:
        today = datetime.now(timezone.utc).date()
        first = datetime.fromisoformat(dates[0]).date()
        if first in (today, today - timedelta(days=1)):
            streak = 1
            for i in range(1, len(dates)):
                prev = datetime.fromisoformat(dates[i - 1]).date()
                cur = datetime.fromisoformat(dates[i]).date()
                if prev - cur == timedelta(days=1):
                    streak += 1
                else:
                    break

    return {
        "videos": row["videos"] or 0,
        "attempts": row["attempts"] or 0,
        "sentences": row["sentences"] or 0,
        "total_minutes": round((row["total_seconds"] or 0) / 60, 1),
        "days": row["days"] or 0,
        "streak": streak,
    }


# timedelta is used in get_stats; import here to avoid top-level dependency issues
from datetime import timedelta  # noqa: E402
