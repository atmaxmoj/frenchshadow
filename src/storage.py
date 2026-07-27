"""Persistent storage for shadow-reader practice attempts.

Each recorded sentence is saved as a row in SQLite; the audio blob is kept on
disk under recordings/ and referenced by path.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
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
    created_at: str


def _init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
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
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_attempts_video ON attempts(video_id);
        CREATE INDEX IF NOT EXISTS idx_attempts_sentence ON attempts(video_id, sentence_idx);
        CREATE INDEX IF NOT EXISTS idx_attempts_created ON attempts(created_at);
        """
    )
    conn.commit()
    return conn


_CONN: sqlite3.Connection | None = None


def _conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        _CONN = _init_db()
    return _CONN


def save_attempt(
    video_id: str,
    sentence_idx: int,
    sentence_text: str,
    language: str,
    audio_bytes: bytes,
    analysis: dict[str, Any],
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
                              overall_score, analysis_json, recording_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            created_at,
        ),
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
        created_at=row["created_at"],
    )


def get_recording_path(attempt_id: str) -> Path | None:
    """Return the filesystem path for an attempt's recording, if it exists."""
    attempt = get_attempt(attempt_id)
    if attempt is None:
        return None
    path = Path(attempt.recording_path)
    return path if path.exists() else None
