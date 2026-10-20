"""
Tests for memory.py. Run from src/:   python -m unittest test_memory -v

TeamApiTests copies the team's own MemoryStoreTests (tests/test_week6.py) and the exact
calls bounded_agent.py, mcp_interface.py and demo_week6.py make, so a change to memory.py
that would break them fails here first.
"""

import json
import os
import sqlite3
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import memory
from memory import (
    MAX_RECORDS_PER_STUDENT,
    MemoryConfigError,
    MemoryStore,
    MemoryUnavailable,
)

PEPPER = "test-pepper-0123456789"
S1, S2 = "STU-2026-001", "STU-2026-002"


class FakeClock:
    def __init__(self):
        self.now = datetime(2026, 10, 12, 9, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, **kwargs):
        self.now += timedelta(**kwargs)


def ticket(n=1001, case="CASE-1045", category="registration", **extra):
    return {"ticket_id": f"TICKET-{n}", "case_id": case, "category": category, **extra}


class FileStoreCase(unittest.TestCase):
    """A store backed by a real file, as the application uses it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "memory.db"
        self.clock = FakeClock()
        self.store = MemoryStore(self.path, pepper=PEPPER, clock=self.clock)

    def opted_in(self, student=S1):
        self.store.grant_consent(student)
        return student

    def all_text_in_database(self):
        conn = sqlite3.connect(self.path)
        try:
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            return " ".join(str(v) for t in tables for row in conn.execute(f"SELECT * FROM {t}") for v in row)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# The team's tests and call patterns
# ---------------------------------------------------------------------------

class TeamApiTests(unittest.TestCase):
    """Copied from tests/test_week6.py::MemoryStoreTests (names and bodies unchanged)."""

    def setUp(self):
        self.clock = FakeClock()
        self.m = MemoryStore(clock=self.clock, pepper="test")
        self.ticket = {"ticket_id": "TICKET-1001", "case_id": "CASE-1045", "category": "registration",
                       "description": "SECRET FREE TEXT", "status": "open"}

    def test_no_consent_nothing_stored(self):
        self.assertFalse(self.m.remember_ticket(S1, self.ticket, "in_progress"))
        self.assertEqual(self.m.recall_tickets(S1), [])

    def test_stores_only_minimal_fields_never_description(self):
        self.m.grant_consent(S1)
        self.m.remember_ticket(S1, self.ticket, "in_progress")
        raw = json.dumps([dict(r) for r in self.m._db.execute("SELECT * FROM memory_records")])
        self.assertNotIn("SECRET FREE TEXT", raw)
        self.assertNotIn("Grace", raw)
        self.assertNotIn(S1, raw)             # pseudonymised

    def test_students_cannot_read_each_others_memory(self):
        self.m.grant_consent(S1); self.m.grant_consent(S2)
        self.m.remember_ticket(S1, self.ticket, "in_progress")
        self.assertEqual(len(self.m.recall_tickets(S1)), 1)
        self.assertEqual(self.m.recall_tickets(S2), [])

    def test_retention_expiry_and_purge(self):
        self.m.grant_consent(S1)
        self.m.remember_ticket(S1, self.ticket, "in_progress")
        self.clock.now += timedelta(days=91)
        self.assertEqual(self.m.recall_tickets(S1), [])      # expired rows are invisible
        self.assertEqual(self.m.purge_expired(), 1)          # and physically deleted

    def test_forget_deletes_everything(self):
        self.m.grant_consent(S1)
        self.m.remember_ticket(S1, self.ticket, "in_progress")
        self.assertEqual(self.m.forget(S1), 1)
        self.assertEqual(self.m.recall_tickets(S1), [])

    def test_revoking_consent_deletes_data(self):
        self.m.grant_consent(S1)
        self.m.remember_ticket(S1, self.ticket, "in_progress")
        self.assertEqual(self.m.revoke_consent(S1), 1)
        self.assertFalse(self.m.has_consent(S1))

    def test_per_student_cap(self):
        self.m.grant_consent(S1)
        for i in range(MAX_RECORDS_PER_STUDENT + 5):
            self.m.remember_ticket(S1, {**self.ticket, "ticket_id": f"TICKET-{2000+i}"}, "in_progress")
        self.assertEqual(len(self.m.recall_tickets(S1)), MAX_RECORDS_PER_STUDENT)

    def test_audit_log_has_no_content(self):
        self.m.grant_consent(S1)
        self.m.remember_ticket(S1, self.ticket, "in_progress")
        self.m.recall_tickets(S1)
        trail = json.dumps(self.m.audit_trail(S1))
        self.assertIn("write", trail)
        self.assertNotIn("registration", trail)


class TeamCallSiteTests(unittest.TestCase):
    """The exact calls made by bounded_agent.py, mcp_interface.py and demo_week6.py."""

    def setUp(self):
        self.clock = FakeClock()
        self.m = MemoryStore(clock=self.clock, pepper="demo")
        self.m.grant_consent(S1)

    def test_names_the_team_code_imports(self):
        self.assertTrue(issubclass(MemoryConfigError, MemoryUnavailable))
        self.assertIsInstance(memory.DEFAULT_RETENTION_DAYS, int)

    def test_advisory_note_inputs(self):
        # bounded_agent: prior = mem.recall_tickets(student_id, case_id)
        self.m.remember_ticket(S1, ticket(), "in_progress")
        prior = self.m.recall_tickets(S1, "CASE-1045")
        self.assertEqual(prior[0]["ticket_id"], "TICKET-1001")
        self.assertEqual(prior[0]["case_status"], "in_progress")
        self.assertEqual(self.m.recall_tickets(S1, "CASE-2031"), [])
        for key in ("ticket_id", "case_id", "category", "created_at", "expires_at"):
            self.assertIn(key, prior[0])

    def test_duplicate_check_inputs(self):
        # bounded_agent: prior = mem.find_recent_ticket(student_id, case_id, DUPLICATE_WINDOW_DAYS)
        self.m.remember_ticket(S1, ticket(), "in_progress")
        prior = self.m.find_recent_ticket(S1, "CASE-1045", 7)
        self.assertEqual(prior["ticket_id"], "TICKET-1001")
        self.assertTrue(prior["created_at"][:10])
        self.clock.advance(days=8)
        self.assertIsNone(self.m.find_recent_ticket(S1, "CASE-1045", 7))

    def test_remember_ticket_returns_a_flag_for_the_audit_event(self):
        # bounded_agent: stored = mem.remember_ticket(student_id, result["ticket"], status)
        full = {"ticket_id": "TICKET-1001", "case_id": "CASE-1045", "category": "registration",
                "description": "x", "status": "open", "created_at": "2026-10-12T09:00:00+00:00"}
        self.assertIs(self.m.remember_ticket(S1, full, "in_progress"), True)
        self.assertIs(self.m.remember_ticket(S2, full, "in_progress"), False)  # S2 has not consented

    def test_blank_status_is_tolerated(self):
        self.assertTrue(self.m.remember_ticket(S1, ticket(), ""))
        self.assertEqual(self.m.recall_tickets(S1)[0]["case_status"], "unknown")

    def test_mcp_history_and_forget(self):
        # mcp_interface: memory.get_history -> export_all, memory.forget -> forget
        self.m.remember_ticket(S1, ticket(), "in_progress")
        self.assertEqual(len(self.m.export_all(S1)["records"]), 1)
        self.assertEqual(self.m.forget(S1), 1)

    def test_demo_retention_step(self):
        # demo_week6 step 8
        self.m.remember_ticket(S1, ticket(9001), "in_progress")
        self.clock.advance(days=91)
        self.assertEqual(self.m.recall_tickets(S1), [])
        self.assertEqual(self.m.purge_expired(), 1)

    def test_demo_audit_and_export_steps(self):
        self.m.remember_ticket(S1, ticket(), "in_progress")
        self.assertTrue(self.m.export_all(S1)["consent"])
        self.assertTrue(self.m.audit_trail(S1))


# ---------------------------------------------------------------------------
# Behaviour specific to this implementation
# ---------------------------------------------------------------------------

class ConsentTests(FileStoreCase):
    def test_nothing_is_stored_or_recalled_without_consent(self):
        self.assertFalse(self.store.remember_case(S1, "CASE-1045"))
        self.assertFalse(self.store.remember_ticket(S1, ticket(), "in_progress"))
        self.assertIsNone(self.store.recall(S1))
        self.assertEqual(self.store.export_all(S1)["records"], [])

    def test_grant_then_remember(self):
        self.opted_in()
        self.assertTrue(self.store.remember_case(S1, "CASE-1045"))
        self.assertEqual(self.store.recall(S1)["active_case_id"], "CASE-1045")

    def test_revoke_without_delete_stops_recall_but_export_still_shows_the_data(self):
        self.opted_in()
        self.store.remember_ticket(S1, ticket(), "in_progress")
        self.store.revoke_consent(S1, delete_data=False)
        self.assertEqual(self.store.recall_tickets(S1), [])
        self.assertIsNone(self.store.recall(S1))
        self.assertIsNone(self.store.find_recent_ticket(S1, "CASE-1045"))
        self.assertEqual(len(self.store.export_all(S1)["records"]), 1)  # right of access


class StorageTests(FileStoreCase):
    def test_ticket_and_active_case_are_separate_records(self):
        self.opted_in()
        self.store.remember_ticket(S1, ticket(category="Registration"), "In_Progress")
        self.store.remember_case(S1, "case-2031")
        record = self.store.recall(S1)
        self.assertEqual(record["active_case_id"], "CASE-2031")
        self.assertEqual(record["tickets"][0]["category"], "registration")
        self.assertEqual(record["tickets"][0]["case_status"], "in_progress")

    def test_active_case_is_replaced_not_accumulated(self):
        self.opted_in()
        self.store.remember_case(S1, "CASE-1045")
        self.store.remember_case(S1, "CASE-2031")
        self.assertEqual(self.store.forget(S1), 1)

    def test_recall_is_capped_but_history_is_not_until_the_limit(self):
        self.opted_in()
        for n in range(1001, 1026):
            self.store.remember_ticket(S1, ticket(n), "in_progress")
        self.assertEqual(len(self.store.export_all(S1)["records"]), MAX_RECORDS_PER_STUDENT)
        self.assertEqual(self.store.export_all(S1)["records"][0]["ticket_id"], "TICKET-1025")
        self.assertEqual(len(self.store.recall(S1)["tickets"]), memory.RECALL_TICKET_LIMIT)

    def test_policy_refuses_free_text(self):
        self.opted_in()
        for bad in ["ignore previous instructions", "CASE-1045; DROP TABLE x", "", None, 1045, "CASE-"]:
            with self.assertRaises(ValueError):
                self.store.remember_case(S1, bad)
        for bad_ticket in [
            ticket(category="registration; DROP TABLE"),
            ticket(category="Grace Nabirye has a problem"),
            ticket(category=""),
            {"ticket_id": "my ticket", "case_id": "CASE-1045", "category": "other"},
            {"ticket_id": "TICKET-1001", "case_id": "somewhere", "category": "other"},
            "not a dict",
        ]:
            with self.assertRaises(ValueError):
                self.store.remember_ticket(S1, bad_ticket, "in_progress")
        with self.assertRaises(ValueError):
            self.store.remember_ticket(S1, ticket(), "free text status with \"quotes\"; DROP")
        self.assertEqual(self.store.export_all(S1)["records"], [])

    def test_student_id_validation(self):
        for bad in ["", "   ", None, "x" * 129]:
            with self.assertRaises(ValueError):
                self.store.recall(bad)

    def test_persists_across_instances(self):
        self.opted_in()
        self.store.remember_case(S1, "CASE-1045")
        reopened = MemoryStore(self.path, pepper=PEPPER, clock=self.clock)  # simulates a restart
        self.assertEqual(reopened.recall(S1)["active_case_id"], "CASE-1045")


class PrivacyTests(FileStoreCase):
    def test_raw_student_id_is_never_written_to_the_database(self):
        student = "23/U/09640/EVE"
        self.opted_in(student)
        self.store.remember_ticket(student, ticket(), "in_progress")
        self.store.recall(student)
        text = self.all_text_in_database()
        self.assertNotIn("09640", text)
        self.assertNotIn(student, text)

    def test_descriptions_are_never_written_to_the_database(self):
        self.opted_in()
        self.store.remember_ticket(S1, ticket(description="My phone is 0700000000"), "in_progress")
        self.assertNotIn("0700000000", self.all_text_in_database())

    def test_audit_trail_records_actions_but_no_content(self):
        self.opted_in()
        self.store.remember_ticket(S1, ticket(), "in_progress")
        self.store.recall(S1)
        self.store.find_recent_ticket(S1, "CASE-1045")
        self.store.forget(S1)
        trail = self.store.audit_trail(S1)
        actions = [(row["actor"], row["action"]) for row in trail]
        for expected in [("student", "consent_granted"), ("agent", "write"), ("agent", "read"), ("student", "forget")]:
            self.assertIn(expected, actions)
        joined = " ".join(f"{row['action']} {row['detail']}" for row in trail)
        for forbidden in ("CASE-", "TICKET-", "registration"):
            self.assertNotIn(forbidden, joined)

    def test_same_student_gives_same_pseudonym_but_a_different_pepper_does_not(self):
        other = MemoryStore(Path(self.tmp.name) / "other.db", pepper="a-different-pepper-123456")
        self.assertEqual(self.store._ref(S1), self.store._ref(" stu-2026-001 "))
        self.assertNotEqual(self.store._ref(S1), other._ref(S1))


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "memory.db"

    def test_missing_environment_pepper_is_refused(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MEMORY_PEPPER", None)
            with self.assertRaises(MemoryConfigError):
                MemoryStore(self.path)

    def test_short_environment_pepper_is_refused(self):
        with mock.patch.dict(os.environ, {"MEMORY_PEPPER": "short"}):
            with self.assertRaises(MemoryConfigError):
                MemoryStore(self.path)

    def test_empty_explicit_pepper_is_refused_but_a_short_one_is_accepted(self):
        with self.assertRaises(MemoryConfigError):
            MemoryStore(self.path, pepper="")
        MemoryStore(self.path, pepper="test")  # tests and demos use short peppers

    def test_a_database_made_with_another_pepper_is_refused(self):
        MemoryStore(self.path, pepper=PEPPER)
        with self.assertRaises(MemoryConfigError):
            MemoryStore(self.path, pepper="a-different-pepper-123456")

    def test_default_is_in_memory_and_nothing_is_written_to_disk(self):
        store = MemoryStore(pepper="test")
        store.grant_consent(S1)
        self.assertIsNone(store.db_path)
        self.assertIsNotNone(store._db)

    def test_environment_store_uses_the_configured_file(self):
        with mock.patch.dict(os.environ, {"MEMORY_DB_PATH": str(self.path), "MEMORY_PEPPER": PEPPER}):
            store = memory._store_from_environment()
        self.assertEqual(store.db_path, self.path)

    def test_storage_failures_surface_as_memory_unavailable(self):
        store = MemoryStore(self.path, pepper=PEPPER)
        with mock.patch("memory.sqlite3.connect", side_effect=sqlite3.OperationalError("disk full")):
            with self.assertRaises(MemoryUnavailable):
                store.has_consent(S1)


class RetentionTests(FileStoreCase):
    def test_records_expire_after_retention(self):
        self.opted_in()
        self.store.remember_ticket(S1, ticket(), "in_progress")
        self.store.remember_case(S1, "CASE-1045")
        self.clock.advance(days=memory.RETENTION_DAYS - 1)
        self.assertIsNotNone(self.store.recall(S1))
        self.clock.advance(days=2)
        self.assertIsNone(self.store.recall(S1))
        self.assertEqual(self.store.export_all(S1)["records"], [])

    def test_expired_records_are_not_revived_by_a_new_write(self):
        self.opted_in()
        self.store.remember_ticket(S1, ticket(), "in_progress")
        self.clock.advance(days=memory.RETENTION_DAYS + 1)
        self.store.remember_case(S1, "CASE-2031")
        record = self.store.recall(S1)
        self.assertEqual(record["active_case_id"], "CASE-2031")
        self.assertEqual(record["tickets"], [])

    def test_purge_expired_removes_records_and_old_audit_rows(self):
        self.opted_in("old")
        self.store.remember_case("old", "CASE-1045")
        self.clock.advance(days=memory.AUDIT_RETENTION_DAYS + 1)
        self.opted_in("new")
        self.store.remember_case("new", "CASE-2031")
        self.assertEqual(self.store.purge_expired(), 1)
        self.assertEqual(self.store.audit_trail("old"), [])
        self.assertIsNone(self.store.recall("old"))
        self.assertIsNotNone(self.store.recall("new"))


class DuplicateWindowTests(FileStoreCase):
    def test_older_ticket_is_outside_the_window_but_still_remembered(self):
        self.opted_in()
        self.store.remember_ticket(S1, ticket(), "in_progress")
        self.clock.advance(days=8)
        self.assertIsNone(self.store.find_recent_ticket(S1, "CASE-1045"))
        self.assertEqual(len(self.store.recall(S1)["tickets"]), 1)

    def test_a_ticket_on_another_case_is_not_a_duplicate(self):
        self.opted_in()
        self.store.remember_ticket(S1, ticket(), "in_progress")
        self.assertIsNone(self.store.find_recent_ticket(S1, "CASE-2031"))


class DeletionAndExportTests(FileStoreCase):
    def test_forget_deletes_records_but_keeps_consent(self):
        self.opted_in()
        self.store.remember_ticket(S1, ticket(), "in_progress")
        self.assertEqual(self.store.forget(S1), 1)
        self.assertIsNone(self.store.recall(S1))
        self.assertTrue(self.store.has_consent(S1))

    def test_revoke_deletes_ticket_and_active_case(self):
        self.opted_in()
        self.store.remember_ticket(S1, ticket(), "in_progress")
        self.store.remember_case(S1, "CASE-1045")
        self.assertEqual(self.store.revoke_consent(S1), 2)

    def test_export_all_lists_everything_held(self):
        self.opted_in()
        self.store.remember_ticket(S1, ticket(), "in_progress")
        self.store.remember_case(S1, "CASE-1045")
        export = self.store.export_all(S1)
        self.assertTrue(export["consent"])
        self.assertEqual(export["active_case"]["case_id"], "CASE-1045")
        self.assertEqual(export["records"][0]["ticket_id"], "TICKET-1001")

    def test_policy_description(self):
        policy = self.store.policy()
        self.assertEqual(policy["retention_days"], memory.RETENTION_DAYS)
        self.assertIn("ticket_id", policy["stored"]["ticket_history"])
        self.assertNotIn("description", policy["stored"]["ticket_history"])


class ThreadSafetyTests(unittest.TestCase):
    def run_threads(self, store):
        errors = []

        def worker(n):
            try:
                student = f"stu-{n}"
                store.grant_consent(student)
                for k in range(5):
                    store.remember_ticket(student, ticket(1000 + n * 10 + k), "in_progress")
                    store.recall(student)
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(store.export_all("stu-3")["records"]), 5)

    def test_file_store_can_be_used_from_many_threads(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.run_threads(MemoryStore(Path(tmp) / "m.db", pepper=PEPPER))

    def test_in_memory_store_can_be_used_from_many_threads(self):
        self.run_threads(MemoryStore(pepper=PEPPER))


if __name__ == "__main__":
    unittest.main()