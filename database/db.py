"""
SQLite Persistent Storage Layer with WAL (Write-Ahead Logging) for genAI.
Replaces volatile in-memory state with ACID-compliant, crash-resilient disk storage.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

DB_DIR = Path(__file__).resolve().parent.parent / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "epistemic_store.db"


def get_connection() -> sqlite3.Connection:
    """Connect to SQLite database with WAL mode enabled for high concurrency."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=30.0)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables with indexes."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS turns (
                turn_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                research TEXT,
                sources_json TEXT,
                deliverables_json TEXT,
                plot_path TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
            CREATE INDEX IF NOT EXISTS idx_turns_created ON turns(created_at);

            CREATE TABLE IF NOT EXISTS uploaded_papers (
                paper_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                page_count INTEGER NOT NULL,
                text TEXT NOT NULL,
                uploaded_at TEXT NOT NULL
            );
        """)


# =====================================================================
# Session & Turn Management
# =====================================================================

def touch_session(session_id: str):
    """Ensure session exists and update timestamp."""
    now = datetime.now().isoformat()
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO sessions (session_id, created_at, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET updated_at = ?
        """, (session_id, now, now, now))


def save_turn(
    session_id: str,
    turn_id: str,
    prompt: str,
    research: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    deliverables: Optional[Dict[str, Any]] = None,
    plot_path: Optional[str] = None
):
    """Persist a completed multi-agent conversational research turn."""
    touch_session(session_id)
    now = datetime.now().isoformat()
    sources_json = json.dumps(sources or [], ensure_ascii=False)
    deliverables_json = json.dumps(deliverables or {}, ensure_ascii=False)

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO turns (turn_id, session_id, prompt, research, sources_json, deliverables_json, plot_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(turn_id) DO UPDATE SET
                research = excluded.research,
                sources_json = excluded.sources_json,
                deliverables_json = excluded.deliverables_json,
                plot_path = excluded.plot_path
        """, (turn_id, session_id, prompt, research, sources_json, deliverables_json, plot_path, now))


def get_session_history(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieve chronologically ordered history of conversational turns for a session."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT turn_id, prompt, research, sources_json, deliverables_json, plot_path, created_at
            FROM turns
            WHERE session_id = ?
            ORDER BY created_at ASC
            LIMIT ?
        """, (session_id, limit)).fetchall()

        history = []
        for r in rows:
            history.append({
                "turn_id": r["turn_id"],
                "prompt": r["prompt"],
                "research": r["research"] or "",
                "sources": json.loads(r["sources_json"] or "[]"),
                "deliverables": json.loads(r["deliverables_json"] or "{}"),
                "plot_path": r["plot_path"],
                "timestamp": r["created_at"]
            })
        return history


def get_last_turn(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve the most recent research turn for a session."""
    with get_connection() as conn:
        row = conn.execute("""
            SELECT turn_id, prompt, research, sources_json, deliverables_json, plot_path, created_at
            FROM turns
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (session_id,)).fetchone()

        if not row:
            return None
        return {
            "turn_id": row["turn_id"],
            "prompt": row["prompt"],
            "research": row["research"] or "",
            "sources": json.loads(row["sources_json"] or "[]"),
            "deliverables": json.loads(row["deliverables_json"] or "{}"),
            "plot_path": row["plot_path"],
            "timestamp": row["created_at"]
        }


def reset_session(session_id: str):
    """Purge all conversation turns for a session."""
    with get_connection() as conn:
        conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))


def get_all_sessions(limit: int = 25) -> List[Dict[str, Any]]:
    """Retrieve recent research sessions with their initial prompt, turn count, and timestamp."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT s.session_id, s.created_at, s.updated_at,
                   COUNT(t.turn_id) as turn_count,
                   (SELECT prompt FROM turns WHERE session_id = s.session_id ORDER BY created_at ASC LIMIT 1) as first_prompt
            FROM sessions s
            JOIN turns t ON s.session_id = t.session_id
            GROUP BY s.session_id
            ORDER BY s.updated_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [
            {
                "session_id": r["session_id"],
                "first_prompt": r["first_prompt"] or "Research Thread",
                "turn_count": r["turn_count"],
                "updated_at": r["updated_at"]
            }
            for r in rows
        ]


# =====================================================================
# Uploaded Reference Papers Management
# =====================================================================

def save_uploaded_paper(paper_id: str, filename: str, page_count: int, text: str):
    """Persist an ingested PDF research paper."""
    now = datetime.now().isoformat()
    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO uploaded_papers (paper_id, filename, page_count, text, uploaded_at)
            VALUES (?, ?, ?, ?, ?)
        """, (paper_id, filename, page_count, text, now))


def get_all_uploaded_papers() -> List[Dict[str, Any]]:
    """Retrieve list of all indexed reference papers."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT paper_id, filename, page_count, uploaded_at, LENGTH(text) as char_count
            FROM uploaded_papers
            ORDER BY uploaded_at DESC
        """).fetchall()

        return [
            {
                "id": r["paper_id"],
                "filename": r["filename"],
                "page_count": r["page_count"],
                "uploaded_at": r["uploaded_at"],
                "char_count": r["char_count"]
            }
            for r in rows
        ]


def get_uploaded_paper_texts(paper_ids: List[str]) -> List[Dict[str, Any]]:
    """Fetch raw text for selected paper IDs."""
    if not paper_ids:
        return []
    placeholders = ",".join(["?"] * len(paper_ids))
    with get_connection() as conn:
        rows = conn.execute(f"""
            SELECT paper_id, filename, page_count, text
            FROM uploaded_papers
            WHERE paper_id IN ({placeholders})
        """, paper_ids).fetchall()

        return [
            {
                "id": r["paper_id"],
                "filename": r["filename"],
                "page_count": r["page_count"],
                "text": r["text"]
            }
            for r in rows
        ]


def delete_uploaded_paper(paper_id: str) -> bool:
    """Delete an uploaded reference paper."""
    with get_connection() as conn:
        res = conn.execute("DELETE FROM uploaded_papers WHERE paper_id = ?", (paper_id,))
        return res.rowcount > 0


# Initialize on import
init_db()
