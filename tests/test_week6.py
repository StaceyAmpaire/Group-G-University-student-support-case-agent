"""Offline tests for Week 6: state model, memory, memory-aware agent controls, MCP-style interface.
No API key or network needed.   Run:  python -m unittest discover -s tests
"""
from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import test_bounded_agent  # noqa: F401  (installs the google-genai stub if the SDK is absent)
import bounded_agent as agent
import tools as project_tools
from state import SessionState, Phase, IllegalTransition
from memory import MemoryStore, MAX_RECORDS_PER_STUDENT
from mcp_interface import CaseSupportServer, Caller

S1, S2 = "STU-2026-001", "STU-2026-002"
TICKET_MSG = "Check CASE-1045 and please create a follow-up support ticket for this."


class Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 5, 9, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


def new_state(student_id, store):
    st = SessionState(student_id=student_id)
    st.memory_enabled = store.has_consent(student_id)
    return st


class StateModelTests(unittest.TestCase):
    def test_legal_path(self):
        st = SessionState()
        st.transition(Phase.CASE_VERIFIED)
        st.transition(Phase.TICKET_CREATED)
        st.transition(Phase.RESPONDED)
        self.assertTrue(st.is_terminal)
        self.assertEqual([t["to"] for t in st.transitions], ["case_verified", "ticket_created", "responded"])

    def test_cannot_create_ticket_straight_from_intake(self):
        with self.assertRaises(IllegalTransition):
            SessionState().transition(Phase.TICKET_CREATED)

    def test_terminal_state_is_final(self):
        st = SessionState()
        st.transition(Phase.RESPONDED)
        with self.assertRaises(IllegalTransition):
            st.transition(Phase.CASE_VERIFIED)

    def test_snapshot_hides_raw_student_id(self):
        snap = SessionState(student_id=S1).snapshot()
        self.assertNotIn(S1, json.dumps(snap))
        self.assertTrue(snap["student_ref"].startswith("stu_"))


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
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


class MemoryAwareAgentTests(unittest.TestCase):
    def setUp(self):
        project_tools.TICKET_STORE.clear()
        self.clock = Clock()
        self.m = MemoryStore(clock=self.clock, pepper="test")
        self.m.grant_consent(S1)

    def _create(self, msg=TICKET_MSG, sid=S1):
        st = new_state(sid, self.m)
        agent.execute_tool("get_case_status", {"case_id": "CASE-1045"}, st, msg, self.m)
        res = agent.execute_tool(
            "create_support_ticket",
            {"case_id": "CASE-1045", "category": "registration", "description": "Registration missing"},
            st, msg, self.m)
        return st, res

    def test_ticket_is_remembered_after_creation(self):
        st, res = self._create()
        self.assertTrue(res["success"])
        self.assertEqual(st.phase, Phase.TICKET_CREATED)
        self.assertEqual(len(self.m.recall_tickets(S1)), 1)

    def test_memory_improves_lookup_with_advisory_note(self):
        self._create()
        st = new_state(S1, self.m)
        res = agent.execute_tool("get_case_status", {"case_id": "CASE-1045"}, st, "status?", self.m)
        self.assertTrue(res["memory"]["advisory_only"])
        self.assertTrue(res["memory"]["prior_tickets"][0]["ticket_id"].startswith("TICKET-"))

    def test_duplicate_is_warned_not_silently_created(self):
        self._create()
        st, res = self._create()                      # second attempt, no override
        self.assertFalse(res["success"])
        self.assertEqual(res["error"], "duplicate_ticket_warning")
        self.assertEqual(st.phase, Phase.AWAITING_CONFIRMATION)
        self.assertEqual(len(project_tools.TICKET_STORE), 1)

    def test_student_can_override_the_warning(self):
        self._create()
        _, res = self._create("Check CASE-1045 and create another support ticket anyway.")
        self.assertTrue(res["success"])
        self.assertEqual(len(project_tools.TICKET_STORE), 2)

    def test_no_consent_means_no_memory_influence(self):
        st = SessionState(student_id=S2)               # S2 never consented
        st.memory_enabled = self.m.has_consent(S2)
        res = agent.execute_tool("get_case_status", {"case_id": "CASE-2031"}, st, "status?", self.m)
        self.assertNotIn("memory", res)
        self.assertEqual(self.m.recall_tickets(S2), [])

    def test_old_ticket_outside_window_does_not_warn(self):
        self._create()
        self.clock.now += timedelta(days=8)
        _, res = self._create()
        self.assertTrue(res["success"])

    def test_memory_outage_degrades_gracefully(self):
        class Broken:
            def has_consent(self, *_): return True
            def find_recent_ticket(self, *_): raise RuntimeError("db down")
            def recall_tickets(self, *_, **__): raise RuntimeError("db down")
            def remember_ticket(self, *_): raise RuntimeError("db down")
        st = SessionState(student_id=S1); st.memory_enabled = True
        agent.execute_tool("get_case_status", {"case_id": "CASE-1045"}, st, TICKET_MSG, Broken())
        res = agent.execute_tool(
            "create_support_ticket",
            {"case_id": "CASE-1045", "category": "registration", "description": "x issue"},
            st, TICKET_MSG, Broken())
        self.assertTrue(res["success"])                # ticket still works, memory skipped
        self.assertTrue(any(e["action"] == "write_failed" for e in st.memory_events))

    def test_student_cannot_look_up_someone_elses_case(self):
        st = new_state(S2, self.m)
        res = agent.execute_tool("get_case_status", {"case_id": "CASE-1045"}, st, "status?", self.m)
        self.assertFalse(res["success"])
        self.assertEqual(st.phase, Phase.CASE_NOT_FOUND)

    def test_memory_cannot_authorise_a_ticket(self):
        """Having history is not permission: explicit request is still required."""
        self._create()
        st = new_state(S1, self.m)
        agent.execute_tool("get_case_status", {"case_id": "CASE-1045"}, st, "status?", self.m)
        res = agent.execute_tool(
            "create_support_ticket",
            {"case_id": "CASE-1045", "category": "registration", "description": "x issue"},
            st, "What is my case status? I already have a ticket, so I am just wondering what happens next.", self.m)
        self.assertFalse(res["success"])
        self.assertIn("explicitly requested", res["error"])


class MCPInterfaceTests(unittest.TestCase):
    def setUp(self):
        project_tools.TICKET_STORE.clear()
        self.m = MemoryStore(pepper="test"); self.m.grant_consent(S1)
        self.srv = CaseSupportServer(self.m)
        self.full = Caller(S1, {"cases:read", "tickets:write", "memory:read", "memory:delete"})
        self.readonly = Caller(S1, {"cases:read"})

    def call(self, name, args, caller=None):
        return self.srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                "params": {"name": name, "arguments": args}}, caller or self.full)

    def test_initialize_and_list(self):
        init = self.srv.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, self.full)
        self.assertIn("tools", init["result"]["capabilities"])
        names = [t["name"] for t in self.srv.handle(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, self.full)["result"]["tools"]]
        self.assertEqual(len(names), 4)

    def test_list_hides_tools_without_scope(self):
        names = [t["name"] for t in self.srv.handle(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, self.readonly)["result"]["tools"]]
        self.assertEqual(names, ["case.get_status"])

    def test_get_status_hides_student_name(self):
        out = self.call("case.get_status", {"case_id": "CASE-1045"})
        body = json.loads(out["result"]["content"][0]["text"])
        self.assertFalse(out["result"]["isError"])
        self.assertNotIn("student_name", body)

    def test_other_students_case_looks_like_not_found(self):
        out = self.call("case.get_status", {"case_id": "CASE-2031"})
        self.assertTrue(out["result"]["isError"])
        self.assertIn("No case found", out["result"]["content"][0]["text"])

    def test_missing_scope_is_forbidden(self):
        out = self.call("ticket.create", {"case_id": "CASE-1045", "category": "registration",
                                          "description": "portal issue", "confirmed_by_student": True}, self.readonly)
        self.assertEqual(out["error"]["code"], -32001)
        self.assertEqual(project_tools.TICKET_STORE, {})

    def test_schema_validation(self):
        bad_args = [
            {},                                                     # missing
            {"case_id": "1045"},                                    # bad format
            {"case_id": "CASE-1045", "extra": 1},                   # unknown arg
            {"case_id": 1045},                                      # wrong type
        ]
        for args in bad_args:
            with self.subTest(args=args):
                self.assertEqual(self.call("case.get_status", args)["error"]["code"], -32602)

    def test_ticket_requires_student_confirmation(self):
        out = self.call("ticket.create", {"case_id": "CASE-1045", "category": "registration",
                                          "description": "portal issue", "confirmed_by_student": False})
        self.assertTrue(out["result"]["isError"])
        self.assertEqual(project_tools.TICKET_STORE, {})

    def test_ticket_rejected_for_resolved_case(self):
        caller = Caller(S2, {"cases:read", "tickets:write"})
        out = self.call("ticket.create", {"case_id": "CASE-2031", "category": "fees",
                                          "description": "PRN issue here", "confirmed_by_student": True}, caller)
        self.assertTrue(out["result"]["isError"])

    def test_ticket_create_then_memory_history_then_forget(self):
        out = self.call("ticket.create", {"case_id": "CASE-1045", "category": "registration",
                                          "description": "portal issue", "confirmed_by_student": True})
        self.assertFalse(out["result"]["isError"])
        hist = json.loads(self.call("memory.get_history", {})["result"]["content"][0]["text"])
        self.assertEqual(len(hist["records"]), 1)
        gone = json.loads(self.call("memory.forget", {"confirm": True})["result"]["content"][0]["text"])
        self.assertEqual(gone["deleted_records"], 1)

    def test_unknown_tool_and_method_and_bad_json(self):
        self.assertEqual(self.call("db.drop_all", {})["error"]["code"], -32601)
        self.assertEqual(self.srv.handle({"jsonrpc": "2.0", "id": 1, "method": "x"}, self.full)["error"]["code"], -32601)
        self.assertEqual(json.loads(self.srv.handle_json("{nope", self.full))["error"]["code"], -32700)

    def test_audit_has_no_arguments_or_content(self):
        self.call("case.get_status", {"case_id": "CASE-1045"})
        self.assertNotIn("CASE-1045", json.dumps(self.srv.audit))


if __name__ == "__main__":
    unittest.main()
