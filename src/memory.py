"""
memory.py

Persistent, consent-based memory for the University Student-Support Case Agent
(Week 6). One memory module serves every caller: bounded_agent.py, mcp_interface.py,
demo_week6.py and agent_loop.py.

Use case: case history
    Remember the tickets this assistant opened for a student, and their most
    recently active case, across sessions. A returning student gets a note about an
    earlier ticket, and the agent can warn about a duplicate opened a few days ago.

Memory policy (fixed in code; the model never reads or writes memory)
    CONSENT          Opt-in. Nothing is stored or recalled for a student until the
                     application calls grant_consent(). The model can never grant
                     consent. revoke_consent() withdraws it and deletes the data.
    WHAT IS STORED   kind='ticket_history' (newest 20 per student): ticket_id,
                     case_id, category, case_status (the status when the ticket was
                     opened), created_at, expires_at.
                     kind='active_case' (one per student): case_id.
                     Every value must match an ID or short-label pattern, so free
                     text, names, messages and instructions cannot be stored.
    NOT STORED       Student names, issue descriptions, grades, fees, model prompts
                     or responses, or any decision. A case's CURRENT status is always
                     re-read with get_case_status, never taken from memory.
    WHY              Saves the student re-entering an ID and prevents accidental
                     duplicate tickets.
    WHO CAN ACCESS   Only the application, using the student_id it supplies after
                     authentication (never one the model supplies). The model never
                     receives a handle to this store. Student IDs are stored only as
                     an HMAC-SHA256 pseudonym keyed by a secret pepper.
    RETENTION        RETENTION_DAYS (90) after a record is written. Expired records are
                     ignored when read and removed by purge_expired(). The duplicate
                     check only looks back DUPLICATE_WINDOW_DAYS (7).
    DELETION         forget(student_id) deletes stored records and keeps consent.
                     revoke_consent(student_id) withdraws consent and deletes records.
                     export_all(student_id) shows everything held (right of access).
    AUDIT            audit_log records who did what and when, never content. Rows older
                     than AUDIT_RETENTION_DAYS are removed by purge_expired().

Memory advises; it never decides alone. The callers show a duplicate warning to the
student, who can override it, and a remembered ID can never authorise an action.

Using the store
    * MemoryStore(db_path=":memory:") is for tests and demos: nothing outlives the
      process. Pass a file path, or use default_store(), for real persistence.
    * default_store() reads MEMORY_PEPPER (a random secret of 16+ characters) and
      MEMORY_DB_PATH (default data/memory.db) from the environment / .env. Keep both
      the pepper and data/ out of git. Changing the pepper later would hide every
      record, so a database created with a different pepper is refused.
    * A pepper passed explicitly (pepper="...") is accepted at any non-empty length so
      tests stay simple; the length rule applies to the environment pepper.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import re
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

load_dotenv()

RETENTION_DAYS = 90
DEFAULT_RETENTION_DAYS = RETENTION_DAYS
DUPLICATE_WINDOW_DAYS = 7
MAX_RECORDS_PER_STUDENT = 20
RECALL_TICKET_LIMIT = 5
AUDIT_RETENTION_DAYS = 365
MIN_ENV_PEPPER_LENGTH = 16
MAX_STUDENT_ID_LENGTH = 128

IN_MEMORY = ":memory:"
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "memory.db"

CASE_ID_PATTERN = re.compile(r"^CASE-\d{1,10}$")
TICKET_ID_PATTERN = re.compile(r"^TICKET-\d{1,10}$")
LABEL_PATTERN = re.compile(r"^[a-z][a-z_]{0,31}$")                # ticket category, e.g. it_support
STATUS_PATTERN = re.compile(r"^[a-z][a-z0-9 _\-]{0,31}$")          # case status, e.g. in_progress

ALLOWED_KINDS = {"active_case", "ticket_history"}
ALLOWED_PAYLOAD_FIELDS = {
    "active_case": {"case_id"},
    "ticket_history": {"ticket_id", "case_id", "category", "case_status"},
}


class MemoryUnavailable(Exception):
    """The store cannot be read or written. Callers should degrade gracefully."""


class MemoryConfigError(MemoryUnavailable):
    """Memory is misconfigured (missing or mismatched pepper, bad path)."""


def _id(value: object, pattern: re.Pattern, field: str) -> str:
    """Memory policy: only ID-shaped values may be stored. Anything else is refused."""
    if not isinstance(value, str) or not pattern.match(value.strip().upper()):
        raise ValueError(f"{field} must look like an ID ({pattern.pattern}); nothing was stored.")
    return value.strip().upper()


def _label(value: object, pattern: re.Pattern, field: str) -> str:
    if not isinstance(value, str) or not pattern.match(value.strip().lower()):
        raise ValueError(f"{field} must be a short label ({pattern.pattern}); nothing was stored.")
    return value.strip().lower()


def _status(value: object) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return "unknown"
    return _label(value, STATUS_PATTERN, "case_status")


def _norm_case(case_id: object) -> str:
    """Reads are lenient: an odd case ID simply matches nothing."""
    return str(case_id).strip().upper()


class MemoryStore:
    def __init__(
        self,
        db_path: os.PathLike | str = IN_MEMORY,
        pepper: Optional[str] = None,
        retention_days: int = RETENTION_DAYS,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        if pepper is None:
            pepper = os.getenv("MEMORY_PEPPER") or ""
            if len(pepper) < MIN_ENV_PEPPER_LENGTH:
                raise MemoryConfigError(
                    f"MEMORY_PEPPER must be set to a secret of at least {MIN_ENV_PEPPER_LENGTH} "
                    "characters (for example: python -c \"import secrets; print(secrets.token_hex(32))\")."
                )
        elif not pepper:
            raise MemoryConfigError("pepper must not be empty.")

        self._pepper = pepper.encode()
        self._retention = timedelta(days=retention_days)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.RLock()

        path = str(db_path) if db_path else IN_MEMORY
        self._in_memory = path == IN_MEMORY
        if self._in_memory:
            # One shared connection for the whole life of the store (used by tests and demos).
            self.db_path = None
            self._db = sqlite3.connect(IN_MEMORY, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
        else:
            # A short-lived connection per operation: safe across threads and releases the
            # file between calls (important on Windows).
            self.db_path = Path(path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # -- internals ---------------------------------------------------------

    @contextlib.contextmanager
    def _tx(self):
        with self._lock:
            conn = None
            try:
                if self._in_memory:
                    conn = self._db
                else:
                    conn = sqlite3.connect(self.db_path, timeout=10)
                    conn.row_factory = sqlite3.Row
                yield conn
                conn.commit()
            except MemoryUnavailable:
                if conn is not None:
                    conn.rollback()
                raise
            except sqlite3.Error as exc:
                if conn is not None:
                    conn.rollback()
                raise MemoryUnavailable(f"memory store error: {exc}") from exc
            except Exception:
                if conn is not None:
                    conn.rollback()
                raise
            finally:
                if conn is not None and not self._in_memory:
                    conn.close()

    def _init_schema(self) -> None:
        with self._tx() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS consent (
                    student_ref TEXT PRIMARY KEY, granted_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS memory_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, student_ref TEXT NOT NULL,
                    kind TEXT NOT NULL, payload TEXT NOT NULL,
                    created_at TEXT NOT NULL, expires_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS idx_mem_student ON memory_records(student_ref);
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
                    "The pepper does not match the one this database was created with. "
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
        return conn.execute("DELETE FROM memory_records WHERE student_ref = ?", (ref,)).rowcount

    def _drop_expired_for(self, conn, ref: str) -> None:
        conn.execute(
            "DELETE FROM memory_records WHERE student_ref = ? AND expires_at <= ?",
            (ref, self._stamp(self._now())),
        )

    def _insert_record(self, conn, ref: str, kind: str, payload: dict) -> None:
        if kind not in ALLOWED_KINDS or set(payload) - ALLOWED_PAYLOAD_FIELDS[kind]:
            raise ValueError("That record is outside the memory policy; nothing was stored.")
        now = self._now()
        conn.execute(
            "INSERT INTO memory_records (student_ref, kind, payload, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (ref, kind, json.dumps(payload, sort_keys=True),
             self._stamp(now), self._stamp(now + self._retention)),
        )

    def _ticket_records(self, conn, ref: str, limit: Optional[int] = None) -> list[dict]:
        sql = (
            "SELECT payload, created_at, expires_at FROM memory_records "
            "WHERE student_ref = ? AND kind = 'ticket_history' AND expires_at > ? ORDER BY id DESC"
        )
        args: list = [ref, self._stamp(self._now())]
        if limit is not None:
            sql += " LIMIT ?"
            args.append(limit)
        records = []
        for row in conn.execute(sql, args):
            record = json.loads(row["payload"])
            record["created_at"] = row["created_at"]
            record["expires_at"] = row["expires_at"]
            records.append(record)
        return records

    def _active_case(self, conn, ref: str) -> Optional[dict]:
        row = conn.execute(
            "SELECT payload, created_at, expires_at FROM memory_records "
            "WHERE student_ref = ? AND kind = 'active_case' AND expires_at > ? ORDER BY id DESC LIMIT 1",
            (ref, self._stamp(self._now())),
        ).fetchone()
        if row is None:
            return None
        return {"case_id": json.loads(row["payload"])["case_id"],
                "updated_at": row["created_at"], "expires_at": row["expires_at"]}

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

    def remember_ticket(self, student_id: str, ticket: dict, case_status: Optional[str] = None) -> bool:
        """
        Store a minimal ticket-history record from a ticket dict (extra keys such as
        description are ignored and never stored). Returns False if there is no consent.
        """
        if not isinstance(ticket, dict):
            raise ValueError("ticket must be a dict; nothing was stored.")
        payload = {
            "ticket_id": _id(ticket.get("ticket_id"), TICKET_ID_PATTERN, "ticket_id"),
            "case_id": _id(ticket.get("case_id"), CASE_ID_PATTERN, "case_id"),
            "category": _label(ticket.get("category"), LABEL_PATTERN, "category"),
            "case_status": _status(case_status),
        }
        ref = self._ref(student_id)
        with self._tx() as conn:
            if not self._has_consent(conn, ref):
                return False
            self._drop_expired_for(conn, ref)
            self._insert_record(conn, ref, "ticket_history", payload)
            conn.execute(
                "DELETE FROM memory_records WHERE student_ref = ? AND kind = 'ticket_history' AND id NOT IN "
                "(SELECT id FROM memory_records WHERE student_ref = ? AND kind = 'ticket_history' "
                "ORDER BY id DESC LIMIT ?)",
                (ref, ref, MAX_RECORDS_PER_STUDENT),
            )
            self._audit(conn, ref, "agent", "write", "kind=ticket_history")
        return True

    def remember_case(self, student_id: str, case_id: str) -> bool:
        """Record the student's most recently active case (one per student). False if no consent."""
        payload = {"case_id": _id(case_id, CASE_ID_PATTERN, "case_id")}
        ref = self._ref(student_id)
        with self._tx() as conn:
            if not self._has_consent(conn, ref):
                return False
            self._drop_expired_for(conn, ref)
            conn.execute("DELETE FROM memory_records WHERE student_ref = ? AND kind = 'active_case'", (ref,))
            self._insert_record(conn, ref, "active_case", payload)
            self._audit(conn, ref, "agent", "write", "kind=active_case")
        return True

    # -- reading ---------------------------------------------------------------------

    def recall_tickets(self, student_id: str, case_id: Optional[str] = None,
                       actor: str = "agent") -> list[dict]:
        """
        Non-expired ticket history for this student, newest first. Empty without consent.
        Each item: ticket_id, case_id, category, case_status, created_at, expires_at.
        """
        ref = self._ref(student_id)
        with self._tx() as conn:
            if not self._has_consent(conn, ref):
                return []
            records = self._ticket_records(conn, ref)
            if case_id is not None:
                wanted = _norm_case(case_id)
                records = [r for r in records if r["case_id"] == wanted]
            self._audit(conn, ref, actor, "read", f"kind=ticket_history n={len(records)}")
        return records

    def recall(self, student_id: str, actor: str = "agent") -> Optional[dict]:
        """
        A compact picture for the agent loop, or None (no consent, or nothing stored):
            {"active_case_id": str | None, "tickets": [newest few], "updated_at": str}
        """
        ref = self._ref(student_id)
        with self._tx() as conn:
            if not self._has_consent(conn, ref):
                return None
            case = self._active_case(conn, ref)
            tickets = self._ticket_records(conn, ref, limit=RECALL_TICKET_LIMIT)
            self._audit(conn, ref, actor, "read",
                        f"active_case={'yes' if case else 'no'} tickets={len(tickets)}")
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
        ref = self._ref(student_id)
        wanted = _norm_case(case_id)
        cutoff = self._stamp(self._now() - timedelta(days=days))
        with self._tx() as conn:
            if not self._has_consent(conn, ref):
                return None
            found = next(
                (r for r in self._ticket_records(conn, ref)
                 if r["case_id"] == wanted and r["created_at"] >= cutoff),
                None,
            )
            self._audit(conn, ref, "agent", "read", f"kind=duplicate_check found={'yes' if found else 'no'}")
        return found

    def export_all(self, student_id: str) -> dict:
        """Everything held about this student (right of access). Works even without consent."""
        ref = self._ref(student_id)
        with self._tx() as conn:
            export = {
                "consent": self._has_consent(conn, ref),
                "records": self._ticket_records(conn, ref),
                "active_case": self._active_case(conn, ref),
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
        with self._tx() as conn:
            deleted = conn.execute(
                "DELETE FROM memory_records WHERE expires_at <= ?", (self._stamp(now),)
            ).rowcount
            conn.execute(
                "DELETE FROM audit_log WHERE at <= ?",
                (self._stamp(now - timedelta(days=AUDIT_RETENTION_DAYS)),),
            )
            self._audit(conn, "*", "system", "purge_expired", f"records_deleted={deleted}")
        return deleted

    # -- audit, documentation, lifecycle ------------------------------------------------

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
                "ticket_history": ["ticket_id", "case_id", "category", "case_status",
                                   "created_at", "expires_at"],
            },
            "not_stored": ["student names", "issue descriptions", "current case status",
                           "model prompts or responses", "decisions"],
            "student_key": "HMAC-SHA256 pseudonym keyed by a secret pepper",
            "retention_days": self._retention.days,
            "duplicate_window_days": DUPLICATE_WINDOW_DAYS,
            "max_ticket_records_per_student": MAX_RECORDS_PER_STUDENT,
            "audit_retention_days": AUDIT_RETENTION_DAYS,
            "deletion": ["forget(student_id)", "revoke_consent(student_id)", "purge_expired()"],
        }

    def close(self) -> None:
        if self._in_memory:
            self._db.close()


def _store_from_environment() -> MemoryStore:
    return MemoryStore(db_path=os.getenv("MEMORY_DB_PATH") or DEFAULT_DB_PATH)


_default_store: Optional[MemoryStore] = None
_default_lock = threading.Lock()


def default_store() -> MemoryStore:
    """The shared, persistent store (created on first use; needs MEMORY_PEPPER)."""
    global _default_store
    with _default_lock:
        if _default_store is None:
            _default_store = _store_from_environment()
        return _default_store