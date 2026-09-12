"""Session history in SQLite.

The point of storing sessions is to answer "am I getting better?", and that
question is only answerable if the comparison is fair. A run at 70% speed with
a guide vocal is not the same test as a cold run at full tempo, so every
session records the conditions it was sung under and the trend view compares
like with like.

Per-note results are stored too, which is what lets the app find the phrases
you get wrong *every time* rather than the ones you happened to fluff today.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..config import HISTORY_DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    song_hash     TEXT    NOT NULL,
    started_at    TEXT    NOT NULL,
    score         REAL    NOT NULL,
    coverage      REAL    NOT NULL,
    mean_cents    REAL    NOT NULL,
    notes_attempted INTEGER NOT NULL,
    notes_total   INTEGER NOT NULL,
    -- conditions: comparing across different ones would be meaningless
    speed         REAL    NOT NULL DEFAULT 1.0,
    transpose     INTEGER NOT NULL DEFAULT 0,
    vocal_gain    REAL    NOT NULL DEFAULT 0.0,
    mode          TEXT    NOT NULL DEFAULT 'quantized',
    tolerance     REAL    NOT NULL DEFAULT 35.0,
    findings_json TEXT    NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS note_results (
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    note_index  INTEGER NOT NULL,
    start       REAL    NOT NULL,
    target_midi REAL    NOT NULL,
    accuracy    REAL    NOT NULL,
    mean_cents  REAL    NOT NULL,
    spread_cents REAL   NOT NULL,
    onset_error REAL,
    PRIMARY KEY (session_id, note_index)
);

CREATE INDEX IF NOT EXISTS idx_sessions_song ON sessions(song_hash, started_at);
CREATE INDEX IF NOT EXISTS idx_note_results_session ON note_results(session_id);
"""


@dataclass
class SessionRow:
    id: int
    song_hash: str
    started_at: str
    score: float
    coverage: float
    speed: float
    transpose: int
    vocal_gain: float
    mode: str

    @property
    def conditions(self) -> str:
        bits = []
        if self.speed != 1.0:
            bits.append(f"{self.speed:.0%} speed")
        if self.transpose:
            bits.append(f"{self.transpose:+d} semitones")
        if self.vocal_gain > 0.05:
            bits.append(f"guide vocal {self.vocal_gain:.0%}")
        return ", ".join(bits) or "full speed, original key, no guide"

    @property
    def when(self) -> datetime:
        return datetime.fromisoformat(self.started_at)


@contextmanager
def connect(path: Path | None = None):
    db = Path(path or HISTORY_DB)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_session(
    song_hash: str,
    scorer,
    *,
    speed: float = 1.0,
    transpose: int = 0,
    vocal_gain: float = 0.0,
    mode: str = "quantized",
    path: Path | None = None,
) -> int:
    summary = scorer.summary()
    with connect(path) as conn:
        cur = conn.execute(
            """INSERT INTO sessions
               (song_hash, started_at, score, coverage, mean_cents,
                notes_attempted, notes_total, speed, transpose, vocal_gain,
                mode, tolerance, findings_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                song_hash,
                scorer.started.isoformat(timespec="seconds"),
                summary["score"],
                summary["coverage"],
                summary["mean_cents"],
                summary["notes_attempted"],
                summary["notes_total"],
                speed,
                transpose,
                vocal_gain,
                mode,
                scorer.tolerance,
                json.dumps(summary["findings"]),
            ),
        )
        session_id = int(cur.lastrowid)
        conn.executemany(
            """INSERT INTO note_results
               (session_id, note_index, start, target_midi, accuracy,
                mean_cents, spread_cents, onset_error)
               VALUES (?,?,?,?,?,?,?,?)""",
            [
                (session_id, n.index, n.start, n.target_midi, n.accuracy,
                 n.mean_cents, n.spread_cents, n.onset_error)
                for n in scorer.attempted
            ],
        )
    return session_id


def recent_sessions(song_hash: str | None = None, limit: int = 25,
                    path: Path | None = None) -> list[SessionRow]:
    with connect(path) as conn:
        if song_hash:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE song_hash=? ORDER BY started_at DESC LIMIT ?",
                (song_hash, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
    return [
        SessionRow(
            id=r["id"], song_hash=r["song_hash"], started_at=r["started_at"],
            score=r["score"], coverage=r["coverage"], speed=r["speed"],
            transpose=r["transpose"], vocal_gain=r["vocal_gain"], mode=r["mode"],
        )
        for r in rows
    ]


def progress_trend(song_hash: str, *, comparable_only: bool = True,
                   path: Path | None = None) -> list[tuple[str, float]]:
    """Score over time for one song.

    ``comparable_only`` restricts to full-speed, original-key runs without a
    guide vocal — otherwise an easier setup would show up as improvement.
    """
    query = "SELECT started_at, score FROM sessions WHERE song_hash=?"
    params: list = [song_hash]
    if comparable_only:
        query += " AND speed=1.0 AND transpose=0 AND vocal_gain<0.05"
    query += " ORDER BY started_at"
    with connect(path) as conn:
        return [(r["started_at"], r["score"]) for r in conn.execute(query, params)]


def persistent_problem_notes(song_hash: str, *, min_sessions: int = 3,
                             limit: int = 10, path: Path | None = None) -> list[dict]:
    """Notes you get wrong repeatedly, not just once.

    A note fluffed in a single run is noise. A note missed across several runs
    is a thing to practise, and that distinction is the whole reason per-note
    results are stored rather than just a session total.
    """
    with connect(path) as conn:
        rows = conn.execute(
            """SELECT nr.note_index, nr.start, nr.target_midi,
                      AVG(nr.accuracy)   AS avg_accuracy,
                      AVG(nr.mean_cents) AS avg_cents,
                      COUNT(*)           AS attempts
               FROM note_results nr
               JOIN sessions s ON s.id = nr.session_id
               WHERE s.song_hash = ?
               GROUP BY nr.note_index
               HAVING attempts >= ?
               ORDER BY avg_accuracy ASC
               LIMIT ?""",
            (song_hash, min_sessions, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id: int, path: Path | None = None) -> None:
    with connect(path) as conn:
        conn.execute("DELETE FROM note_results WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
