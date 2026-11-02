"""
SQLite Persistent Storage Layer with WAL (Write-Ahead Logging) for genAI.
Multi-tenant architecture: Each user receives their own dedicated, isolated SQLite database file.
"""

import sqlite3
import json
import hashlib
import secrets
import uuid
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

DB_DIR = Path(__file__).resolve().parent.parent / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)
MASTER_DB_PATH = DB_DIR / "epistemic_store.db"

USER_DBS_DIR = DB_DIR / "users"
USER_DBS_DIR.mkdir(parents=True, exist_ok=True)


def get_user_db_path(user_id: Optional[str] = None) -> Path:
    """Return path to user's dedicated SQLite database file."""
    if not user_id or str(user_id).strip() in ("", "guest", "undefined", "null"):
        return USER_DBS_DIR / "guest.db"
    clean_id = re.sub(r'[^a-zA-Z0-9_\-]', '', str(user_id).strip())
    return USER_DBS_DIR / f"{clean_id}.db"


def init_user_db(db_path: Path, profile_data: Optional[Dict[str, Any]] = None):
    """Provision a brand new isolated SQLite database file for a user with all tables and indexes."""
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30.0)
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS profile (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                password_hash TEXT,
                plan TEXT NOT NULL DEFAULT 'Pro',
                created_at TEXT NOT NULL,
                last_login TEXT
            );

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

        if profile_data:
            conn.execute("""
                INSERT INTO profile (user_id, name, email, password_hash, plan, created_at, last_login)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    name = excluded.name,
                    email = excluded.email,
                    plan = excluded.plan,
                    last_login = excluded.last_login
            """, (
                profile_data.get("user_id"),
                profile_data.get("name"),
                profile_data.get("email"),
                profile_data.get("password_hash"),
                profile_data.get("plan", "Pro"),
                profile_data.get("created_at", datetime.now().isoformat()),
                profile_data.get("last_login", datetime.now().isoformat())
            ))
            conn.commit()
    finally:
        conn.close()


def get_connection(user_id: Optional[str] = None) -> sqlite3.Connection:
    """Connect to user's dedicated SQLite database (or master DB if user_id is None)."""
    if user_id is None:
        db_path = MASTER_DB_PATH
    else:
        db_path = get_user_db_path(user_id)
        if not db_path.exists():
            init_user_db(db_path)

    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30.0)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize master user registry and seed default accounts."""
    with get_connection(None) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                plan TEXT NOT NULL DEFAULT 'Pro',
                db_file TEXT,
                created_at TEXT NOT NULL,
                last_login TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

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
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS uploaded_papers (
                paper_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                page_count INTEGER NOT NULL,
                text TEXT NOT NULL,
                uploaded_at TEXT NOT NULL
            );
        """)
        # Migration: Ensure all columns exist in master users table
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
        if "db_file" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN db_file TEXT")
        if "last_login" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN last_login TEXT")

        seed_default_users(conn)

    # Also provision guest database
    guest_path = get_user_db_path("guest")
    if not guest_path.exists():
        init_user_db(guest_path)


# =====================================================================
# Session & Turn Management (Per-User Isolated)
# =====================================================================

def touch_session(session_id: str, user_id: Optional[str] = None):
    """Ensure session exists in user's isolated database."""
    now = datetime.now().isoformat()
    with get_connection(user_id) as conn:
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
    plot_path: Optional[str] = None,
    user_id: Optional[str] = None
):
    """Persist a research turn into user's dedicated SQLite database."""
    touch_session(session_id, user_id)
    now = datetime.now().isoformat()
    sources_json = json.dumps(sources or [], ensure_ascii=False)
    deliverables_json = json.dumps(deliverables or {}, ensure_ascii=False)

    with get_connection(user_id) as conn:
        conn.execute("""
            INSERT INTO turns (turn_id, session_id, prompt, research, sources_json, deliverables_json, plot_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(turn_id) DO UPDATE SET
                research = excluded.research,
                sources_json = excluded.sources_json,
                deliverables_json = excluded.deliverables_json,
                plot_path = excluded.plot_path
        """, (turn_id, session_id, prompt, research, sources_json, deliverables_json, plot_path, now))


def get_session_history(session_id: str, limit: int = 10, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve chronologically ordered history of conversational turns for a session from user's DB."""
    with get_connection(user_id) as conn:
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


def get_last_turn(session_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve the most recent research turn for a session from user's DB."""
    with get_connection(user_id) as conn:
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


def reset_session(session_id: str, user_id: Optional[str] = None):
    """Purge all conversation turns for a session from user's DB."""
    with get_connection(user_id) as conn:
        conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))


def get_all_sessions(limit: int = 25, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve recent research sessions from user's isolated database."""
    with get_connection(user_id) as conn:
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
# Uploaded Reference Papers Management (Per-User Isolated)
# =====================================================================

def save_uploaded_paper(paper_id: str, filename: str, page_count: int, text: str, user_id: Optional[str] = None):
    """Persist an ingested PDF research paper in user's isolated database."""
    now = datetime.now().isoformat()
    with get_connection(user_id) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO uploaded_papers (paper_id, filename, page_count, text, uploaded_at)
            VALUES (?, ?, ?, ?, ?)
        """, (paper_id, filename, page_count, text, now))


def get_all_uploaded_papers(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve list of all indexed reference papers for the user."""
    with get_connection(user_id) as conn:
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


def get_uploaded_paper_texts(paper_ids: List[str], user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch raw text for selected paper IDs from user's DB."""
    if not paper_ids:
        return []
    placeholders = ",".join(["?"] * len(paper_ids))
    with get_connection(user_id) as conn:
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


def delete_uploaded_paper(paper_id: str, user_id: Optional[str] = None) -> bool:
    """Delete an uploaded reference paper from user's DB."""
    with get_connection(user_id) as conn:
        res = conn.execute("DELETE FROM uploaded_papers WHERE paper_id = ?", (paper_id,))
        return res.rowcount > 0


# =====================================================================
# User Authentication & Dedicated Database Provisioning
# =====================================================================

def hash_password(password: str, salt: Optional[str] = None) -> str:
    """Hash password with sha256 and unique 16-byte cryptographic salt."""
    if not salt:
        salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return f"{salt}${hashed}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored salt$hash string."""
    try:
        salt, _ = stored_hash.split("$", 1)
        return hash_password(password, salt) == stored_hash
    except Exception:
        return False


def seed_default_users(conn: sqlite3.Connection):
    """Seed initial demo account Zeeshan Rana and provision his dedicated database file."""
    row = conn.execute("SELECT user_id, name, email, password_hash, plan, created_at FROM users WHERE LOWER(email) = 'zeeshan@example.com'").fetchone()
    now = datetime.now().isoformat()
    if not row:
        user_id = str(uuid.uuid4())
        pwd_hash = hash_password("antigravity")
        db_file_name = f"{user_id}.db"
        conn.execute("""
            INSERT INTO users (user_id, name, email, password_hash, plan, db_file, created_at, last_login)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, "Zeeshan Rana", "zeeshan@example.com", pwd_hash, "Pro", db_file_name, now, now))
        profile_data = {
            "user_id": user_id,
            "name": "Zeeshan Rana",
            "email": "zeeshan@example.com",
            "password_hash": pwd_hash,
            "plan": "Pro",
            "created_at": now,
            "last_login": now
        }
    else:
        user_id = row["user_id"]
        profile_data = {
            "user_id": user_id,
            "name": row["name"],
            "email": row["email"],
            "password_hash": row["password_hash"],
            "plan": row["plan"],
            "created_at": row["created_at"],
            "last_login": now
        }

    # Ensure dedicated DB exists on disk
    user_db_path = get_user_db_path(user_id)
    if not user_db_path.exists():
        init_user_db(user_db_path, profile_data)


def create_user(name: str, email: str, password: str, plan: str = "Pro") -> Dict[str, Any]:
    """
    Register a new user account:
    1. Validates unique credentials in master registry
    2. CREATES A BRAND NEW ISOLATED SQLITE DATABASE FILE FOR THIS USER
    3. Saves user profile, password hash, plan, and initial state inside their private database
    """
    name_clean = name.strip()
    email_clean = email.strip().lower()

    if not name_clean:
        raise ValueError("Full name is required.")
    if not email_clean or "@" not in email_clean:
        raise ValueError("A valid email address is required.")
    if not password or len(password) < 4:
        raise ValueError("Password must be at least 4 characters long.")

    with get_connection(None) as master_conn:
        existing = master_conn.execute("SELECT user_id FROM users WHERE LOWER(email) = ?", (email_clean,)).fetchone()
        if existing:
            raise ValueError("An account with this email address already exists.")

        user_id = str(uuid.uuid4())
        pwd_hash = hash_password(password)
        now = datetime.now().isoformat()
        db_file_name = f"{user_id}.db"

        # 1. Store in master users registry
        master_conn.execute("""
            INSERT INTO users (user_id, name, email, password_hash, plan, db_file, created_at, last_login)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, name_clean, email_clean, pwd_hash, plan, db_file_name, now, now))

    # 2. CREATE AND INITIALIZE THE BRAND NEW DEDICATED DATABASE FILE
    user_db_path = get_user_db_path(user_id)
    profile_data = {
        "user_id": user_id,
        "name": name_clean,
        "email": email_clean,
        "password_hash": pwd_hash,
        "plan": plan,
        "created_at": now,
        "last_login": now
    }
    init_user_db(user_db_path, profile_data)

    print(f"[Multi-Tenant Engine] Created brand new user database: {user_db_path} for {email_clean}")

    return {
        "user_id": user_id,
        "name": name_clean,
        "email": email_clean,
        "plan": plan,
        "db_file": db_file_name
    }


def authenticate_user(email: str, password: str) -> Dict[str, Any]:
    """Authenticate user credentials and ensure their dedicated SQLite database is connected."""
    email_clean = email.strip().lower()
    if not email_clean or not password:
        raise ValueError("Email and password are required.")

    with get_connection(None) as master_conn:
        row = master_conn.execute("SELECT * FROM users WHERE LOWER(email) = ?", (email_clean,)).fetchone()
        if not row:
            raise ValueError("Account not found. Please check your email or sign up first.")

        if not verify_password(password, row["password_hash"]):
            raise ValueError("Incorrect password. Please try again.")

        user_id = row["user_id"]
        now = datetime.now().isoformat()
        master_conn.execute("UPDATE users SET last_login = ? WHERE user_id = ?", (now, user_id))

    # Ensure their private DB exists and is updated with last login
    user_db_path = get_user_db_path(user_id)
    profile_data = {
        "user_id": user_id,
        "name": row["name"],
        "email": row["email"],
        "password_hash": row["password_hash"],
        "plan": row["plan"],
        "created_at": row["created_at"],
        "last_login": now
    }
    init_user_db(user_db_path, profile_data)

    return {
        "user_id": user_id,
        "name": row["name"],
        "email": row["email"],
        "plan": row["plan"],
        "db_file": f"{user_id}.db"
    }


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve user public details by ID from master DB."""
    with get_connection(None) as conn:
        row = conn.execute("SELECT user_id, name, email, plan, db_file, created_at, last_login FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if not row:
            return None
        return dict(row)


def get_user_profile_from_db(user_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve user details directly from their own dedicated private database."""
    with get_connection(user_id) as conn:
        row = conn.execute("SELECT user_id, name, email, plan, created_at, last_login FROM profile WHERE user_id = ?", (user_id,)).fetchone()
        if not row:
            return None
        return dict(row)


# Initialize on import
init_db()
