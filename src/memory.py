"""
memory.py

Persistent, consent-based memory for the University Student-Support Case Agent
(Week 6).

Use case: case history
    Remember a student's most recently active case and the tickets this assistant
    opened for them, across conversations and restarts. This lets a returning
    student ask "what's the status of my case?" without retyping the ID, and lets
    the agent warn about a duplicate ticket the student opened days ago.

Memory policy (fixed in code; the model never reads or writes memory)
    CONSENT          Opt-in. Nothing is stored or recalled for a student until the
                     application calls grant_consent(). The model can never grant
                     consent. revoke_consent() stops memory and deletes the data.
    WHAT IS STORED   active_case: one CASE-<digits> ID. ticket_history (newest 20):
                     ticket_id, case_id, category, created_at, expires_at. Values
                     must match ID or short-label patterns, so free text, names,
                     messages and instructions cannot be stored.
    NOT STORED       Message text, names, ticket descriptions, case status or
                     summaries, grades, fees, model output or any decision. A case's
                     status is always re-read with get_case_status, never from memory.
    WHY              Saves re-entering an ID and prevents accidental duplicate
                     tickets. Pointers instead of facts mean memory cannot go stale.
    WHO CAN ACCESS   Only the application, using the student_id it supplies from the
                     logged-in session (never one the model supplies). No tool gives
                     the model access. Student IDs are stored only as an HMAC-SHA256
                     pseudonym keyed by the secret MEMORY_PEPPER.
    RETENTION        RETENTION_DAYS (30) after a record is written. Expired records
                     are ignored when read and removed by purge_expired(). The
                     duplicate-ticket check only looks back DUPLICATE_WINDOW_DAYS (7).
    DELETION         forget(student_id) deletes the student's stored records and keeps
                     their consent. revoke_consent(student_id) withdraws consent and
                     deletes the records. export_all(student_id) shows everything held.
    AUDIT            audit_log records who did what and when, never content. Audit
                     rows older than AUDIT_RETENTION_DAYS are removed by purge_expired().

Memory advises; it never decides alone. See agent_loop.py: a remembered case ID can
never open a ticket, and a duplicate warning can be overridden by the student.

Operational notes
    * Set MEMORY_PEPPER in .env (a random secret of 16+ characters) and keep it out
      of git. Changing it later makes existing records unreachable, so this module
      refuses to open a database that was created with a different pepper.
    * The database is a file (MEMORY_DB_PATH, default data/memory.db). Keep data/ out
      of git.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import os
import re
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

load_dotenv()

RETENTION_DAYS = 30
DUPLICATE_WINDOW_DAYS = 7
MAX_TICKET_RECORDS_PER_STUDENT = 20
RECALL_TICKET_LIMIT = 5
AUDIT_RETENTION_DAYS = 365
MIN_PEPPER_LENGTH = 16
MAX_STUDENT_ID_LENGTH = 128

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "memory.db"

CASE_ID_PATTERN = re.compile(r"^CASE-\d{1,10}$")
TICKET_ID_PATTERN = re.compile(r"^TICKET-\d{1,10}$")
LABEL_PATTERN = re.compile(r"^[a-z][a-z_]{0,31}$")  # ticket category, e.g. it_support


class MemoryConfigError(Exception):
    """Memory is misconfigured (missing or mismatched MEMORY_PEPPER, bad path)."""


def _id(value: object, pattern: re.Pattern, field: str) -> str:
    """Memory policy: only ID-shaped values may be stored. Anything else is refused."""
    if not isinstance(value, str) or not pattern.match(value.strip().upper()):
        raise ValueError(f"{field} must look like an ID ({pattern.pattern}); nothing was stored.")
    return value.strip().upper()


def _label(value: object, field: str) -> str:
    if not isinstance(value, str) or not LABEL_PATTERN.match(value.strip().lower()):
        raise ValueError(f"{field} must be a short label ({LABEL_PATTERN.pattern}); nothing was stored.")
    return value.strip().lower()


class MemoryStore:
    def __init__(
        self,
        db_path: Optional[os.PathLike] = None,
        pepper: Optional[str] = None,
        retention_days: int = RETENTION_DAYS,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        pepper = pepper if pepper is not None else os.getenv("MEMORY_PEPPER")
        if not pepper or len(pepper) < MIN_PEPPER_LENGTH:
            raise MemoryConfigError(
                f"MEMORY_PEPPER must be set to a secret of at least {MIN_PEPPER_LENGTH} characters "
                "(for example: python -c \"import secrets; print(secrets.token_hex(32))\")."
            )
        path = db_path or os.getenv("MEMORY_DB_PATH") or DEFAULT_DB_PATH
        if str(path) == ":memory:":
            raise MemoryConfigError("Persistent memory needs a database file, not ':memory:'.")

        self._pepper = pepper.encode()
        self._retention = timedelta(days=retention_days)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.RLock()
        self.db_path = Path(path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # -- internals ---------------------------------------------------------

    @contextlib.contextmanager
    def _tx(self):
        """One short-lived connection per operation, so the store is safe across threads."""
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _init_schema(self) -> None:
        with self._tx() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS consent (
                    student_ref TEXT PRIMARY KEY, granted_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS active_case (
                    student_ref TEXT PRIMARY KEY, case_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL, expires_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS ticket_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, student_ref TEXT NOT NULL,
                    ticket_id TEXT NOT NULL, case_id TEXT NOT NULL, category TEXT NOT NULL,
                    created_at TEXT NOT NULL, expires_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS idx_ticket_student ON ticket_history(student_ref);
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL,
                    student_ref TEXT NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '');
                """
            )
            # A different pepper would silently hide every record, so refuse to continue.
            fingerprint = hmac.new(self._pepper, b"memory-pepper-check", hashlib.sha256).hexdigest()[:16]
            row = conn.execute("SELECT value FROM meta WHERE key = 'pepper_check'").fetchone()
            if row is None:
                conn.execute("INSERT INTO meta (key, value) VALUES ('pepper_check', ?)", (fingerprint,))
            elif row["value"] != fingerprint:
                raise MemoryConfigError(
                    "MEMORY_PEPPER does not match the one this database was created with. "
                    "Restore the original pepper; records cannot be read without it."
                )

    def _ref(self, student_id: object) -> str:
        if not isinstance(student_id, str) or not student_id.strip():
            raise ValueError("student_id is required.")
        if len(student_id) > MAX_STUDENT_ID_LENGTH:
            raise ValueError("student_id is too long.")
        return hmac.new(self._pepper, student_id.strip().upper().encode(), hashlib.sha256).hexdigest()[:32]

    def _now(self) -> datetime:
        return self._clock().astimezone(timezone.utc)

    @staticmethod
    def _stamp(moment: datetime) -> str:
        return moment.isoformat(timespec="seconds")

    def _audit(self, conn, ref: str, actor: str, action: str, detail: str = "") -> None:
        """Audit rows say who did what and when. They never hold memory content."""
        conn.execute(
            "INSERT INTO audit_log (at, student_ref, actor, action, detail) VALUES (?, ?, ?, ?, ?)",
            (self._stamp(self._now()), ref, actor, action, detail),
        )

    @staticmethod
    def _has_consent(conn, ref: str) -> bool:
        return conn.execute("SELECT 1 FROM consent WHERE student_ref = ?", (ref,)).fetchone() is not None

    @staticmethod
    def _delete_records(conn, ref: str) -> int:
        deleted = conn.execute("DELETE FROM active_case WHERE student_ref = ?", (ref,)).rowcount
        deleted += conn.execute("DELETE FROM ticket_history WHERE student_ref = ?", (ref,)).rowcount
        return deleted

    def _drop_expired_for(self, conn, ref: str) -> None:
        now = self._stamp(self._now())
        conn.execute("DELETE FROM active_case WHERE student_ref = ? AND expires_at <= ?", (ref, now))
        conn.execute("DELETE FROM ticket_history WHERE student_ref = ? AND expires_at <= ?", (ref, now))

    def _set_active_case(self, conn, ref: str, case_id: str) -> None:
        now = self._now()
        conn.execute(
            """
            INSERT INTO active_case (student_ref, case_id, updated_at, expires_at) VALUES (?, ?, ?, ?)
            ON CONFLICT(student_ref) DO UPDATE SET
                case_id = excluded.case_id,
                updated_at = excluded.updated_at,
                expires_at = excluded.expires_at
            """,
            (ref, case_id, self._stamp(now), self._stamp(now + self._retention)),
        )

    # -- consent (granted by the application on the student's say-so, never by the model) --

    def grant_consent(self, student_id: str) -> None:
        ref = self._ref(student_id)
        with self._tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO consent (student_ref, granted_at) VALUES (?, ?)",
                (ref, self._stamp(self._now())),
            )
            self._audit(conn, ref, "student", "consent_granted")

    def has_consent(self, student_id: str) -> bool:
        ref = self._ref(student_id)
        with self._tx() as conn:
            return self._has_consent(conn, ref)

    def revoke_consent(self, student_id: str, delete_data: bool = True) -> int:
        """Withdraw consent. By default also deletes everything stored. Returns records deleted."""
        ref = self._ref(student_id)
        with self._tx() as conn:
            conn.execute("DELETE FROM consent WHERE student_ref = ?", (ref,))
            deleted = self._delete_records(conn, ref) if delete_data else 0
            self._audit(conn, ref, "student", "consent_revoked", f"records_deleted={deleted}")
        return deleted

    # -- writing (called by the application with trusted tool results) ----------

    def remember_case(self, student_id: str, case_id: str) -> bool:
        """Record the student's most recently active case. False if no consent."""
        case_id = _id(case_id, CASE_ID_PATTERN, "case_id")
        ref = self._ref(student_id)
        with self._tx() as conn:
            if not self._has_consent(conn, ref):
                return False
            self._drop_expired_for(conn, ref)
            self._set_active_case(conn, ref, case_id)
            self._audit(conn, ref, "agent", "write", "kind=active_case")
        return True

    def remember_ticket(self, student_id: str, ticket_id: str, case_id: str, category: str) -> bool:
        """Record a ticket opened for the student; its case becomes the active case. False if no consent."""
        ticket_id = _id(ticket_id, TICKET_ID_PATTERN, "ticket_id")
        case_id = _id(case_id, CASE_ID_PATTERN, "case_id")
        category = _label(category, "category")
        ref = self._ref(student_id)
        with self._tx() as conn:
            if not self._has_consent(conn, ref):
                return False
            self._drop_expired_for(conn, ref)
            now = self._now()
            conn.execute(
                "INSERT INTO ticket_history (student_ref, ticket_id, case_id, category, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ref, ticket_id, case_id, category, self._stamp(now), self._stamp(now + self._retention)),
            )
            conn.execute(
                "DELETE FROM ticket_history WHERE student_ref = ? AND id NOT IN "
                "(SELECT id FROM ticket_history WHERE student_ref = ? ORDER BY id DESC LIMIT ?)",
                (ref, ref, MAX_TICKET_RECORDS_PER_STUDENT),
            )
            self._set_active_case(conn, ref, case_id)
            self._audit(conn, ref, "agent", "write", "kind=ticket_history")
        return True

    # -- reading ---------------------------------------------------------------------

    def recall(self, student_id: str, actor: str = "agent") -> Optional[dict]:
        """
        What is remembered for this student, or None (no consent, nothing stored, or expired):
            {"active_case_id": str | None,
             "tickets": [{"ticket_id", "case_id", "category", "created_at"}, ...],  # newest first
             "updated_at": str}
        """
        ref = self._ref(student_id)
        now = self._stamp(self._now())
        with self._tx() as conn:
            if not self._has_consent(conn, ref):
                return None
            case = conn.execute(
                "SELECT case_id, updated_at FROM active_case WHERE student_ref = ? AND expires_at > ?",
                (ref, now),
            ).fetchone()
            tickets = [
                dict(row)
                for row in conn.execute(
                    "SELECT ticket_id, case_id, category, created_at FROM ticket_history "
                    "WHERE student_ref = ? AND expires_at > ? ORDER BY id DESC LIMIT ?",
                    (ref, now, RECALL_TICKET_LIMIT),
                )
            ]
            self._audit(conn, ref, actor, "read", f"active_case={'yes' if case else 'no'} tickets={len(tickets)}")
        if case is None and not tickets:
            return None
        stamps = ([case["updated_at"]] if case else []) + [t["created_at"] for t in tickets]
        return {
            "active_case_id": case["case_id"] if case else None,
            "tickets": tickets,
            "updated_at": max(stamps),
        }

    def find_recent_ticket(self, student_id: str, case_id: str,
                           days: int = DUPLICATE_WINDOW_DAYS) -> Optional[dict]:
        """The newest ticket opened for this case within `days`, if any (for duplicate warnings)."""
        case_id = _id(case_id, CASE_ID_PATTERN, "case_id")
        ref = self._ref(student_id)
        now = self._now()
        with self._tx() as conn:
            if not self._has_consent(conn, ref):
                return None
            row = conn.execute(
                "SELECT ticket_id, case_id, category, created_at FROM ticket_history "
                "WHERE student_ref = ? AND case_id = ? AND expires_at > ? AND created_at >= ? "
                "ORDER BY id DESC LIMIT 1",
                (ref, case_id, self._stamp(now), self._stamp(now - timedelta(days=days))),
            ).fetchone()
            self._audit(conn, ref, "agent", "read", f"kind=duplicate_check found={'yes' if row else 'no'}")
        return dict(row) if row else None

    def export_all(self, student_id: str) -> dict:
        """Everything held about this student (right of access). Works even without consent."""
        ref = self._ref(student_id)
        now = self._stamp(self._now())
        with self._tx() as conn:
            case = conn.execute(
                "SELECT case_id, updated_at, expires_at FROM active_case WHERE student_ref = ? AND expires_at > ?",
                (ref, now),
            ).fetchone()
            tickets = [
                dict(row)
                for row in conn.execute(
                    "SELECT ticket_id, case_id, category, created_at, expires_at FROM ticket_history "
                    "WHERE student_ref = ? AND expires_at > ? ORDER BY id DESC",
                    (ref, now),
                )
            ]
            export = {
                "consent": self._has_consent(conn, ref),
                "active_case": dict(case) if case else None,
                "ticket_history": tickets,
            }
            self._audit(conn, ref, "student", "export")
        return export

    # -- deletion --------------------------------------------------------------------

    def forget(self, student_id: str) -> int:
        """Delete the student's stored records (consent is kept). Returns records deleted."""
        ref = self._ref(student_id)
        with self._tx() as conn:
            deleted = self._delete_records(conn, ref)
            self._audit(conn, ref, "student", "forget", f"records_deleted={deleted}")
        return deleted

    def purge_expired(self) -> int:
        """Delete expired records, and audit rows past AUDIT_RETENTION_DAYS. Returns records deleted."""
        now = self._now()
        stamp = self._stamp(now)
        with self._tx() as conn:
            deleted = conn.execute("DELETE FROM active_case WHERE expires_at <= ?", (stamp,)).rowcount
            deleted += conn.execute("DELETE FROM ticket_history WHERE expires_at <= ?", (stamp,)).rowcount
            conn.execute(
                "DELETE FROM audit_log WHERE at <= ?",
                (self._stamp(now - timedelta(days=AUDIT_RETENTION_DAYS)),),
            )
            self._audit(conn, "*", "system", "purge_expired", f"records_deleted={deleted}")
        return deleted

    # -- audit and documentation --------------------------------------------------------

    def audit_trail(self, student_id: str) -> list[dict]:
        ref = self._ref(student_id)
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT at, actor, action, detail FROM audit_log WHERE student_ref = ? ORDER BY id", (ref,)
            ).fetchall()
        return [dict(row) for row in rows]

    def policy(self) -> dict:
        """The fixed memory policy, for the Memory Design and Data Handling Note."""
        return {
            "consent": "opt-in, granted by the application; revoking deletes the data",
            "stored": {
                "active_case": ["case_id"],
                "ticket_history": ["ticket_id", "case_id", "category", "created_at", "expires_at"],
            },
            "not_stored": ["message text", "names", "ticket descriptions", "case status", "model output"],
            "student_key": "HMAC-SHA256 pseudonym keyed by MEMORY_PEPPER",
            "retention_days": self._retention.days,
            "duplicate_window_days": DUPLICATE_WINDOW_DAYS,
            "max_ticket_records_per_student": MAX_TICKET_RECORDS_PER_STUDENT,
            "audit_retention_days": AUDIT_RETENTION_DAYS,
            "deletion": ["forget(student_id)", "revoke_consent(student_id)", "purge_expired()"],
        }


_default_store: Optional[MemoryStore] = None
_default_lock = threading.Lock()


def default_store() -> MemoryStore:
    """The shared store used by the agent loop (created on first use; needs MEMORY_PEPPER)."""
    global _default_store
    with _default_lock:
        if _default_store is None:
            _default_store = MemoryStore()
        return _default_store
