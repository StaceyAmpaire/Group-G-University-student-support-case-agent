"""
Week 6 demonstration: explicit state + consent-based memory + MCP-style interface.

Runs OFFLINE (no API key, no network): it drives the application layer
(execute_tool / SessionState / MemoryStore / CaseSupportServer) directly, using the
same tool calls the model would request. This isolates what Week 6 adds. The live
model path is run_bounded_agent(message, student_id=..., memory=...).

    python src/demo_week6.py            -> prints the demo, writes docs/evidence/week6_demo_output.json
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import bounded_agent as agent
import tools
from memory import MemoryStore
from state import SessionState
from mcp_interface import CaseSupportServer, Caller

STUDENT = "STU-2026-001"
OTHER = "STU-2026-002"
log = []


def show(title, **data):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)
    for k, v in data.items():
        print(f"{k}: {json.dumps(v, indent=2) if isinstance(v, (dict, list)) else v}")
    log.append({"step": title, **data})


class Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 5, 9, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


def session(mem, sid, message, create=False):
    """One agent 'run': lookup, optionally a ticket request. Returns (state, results)."""
    st = SessionState(student_id=sid)
    st.memory_enabled = mem.has_consent(sid)
    out = {"lookup": agent.execute_tool("get_case_status", {"case_id": "CASE-1045"}, st, message, mem)}
    if create:
        out["ticket"] = agent.execute_tool(
            "create_support_ticket",
            {"case_id": "CASE-1045", "category": "registration",
             "description": "Course registration is not reflecting on my portal."},
            st, message, mem)
    st.transition(__import__("state").Phase.RESPONDED, "demo run complete")
    return st, out


clock = Clock()
mem = MemoryStore(clock=clock, pepper="demo")

# 1 ---------------------------------------------------------------- no consent
st, out = session(mem, STUDENT, "Check CASE-1045 and create a support ticket for it.", create=True)
show("1. NO CONSENT: ticket works, nothing is remembered",
     ticket=out["ticket"].get("ticket", {}).get("ticket_id"),
     stored_records=mem.recall_tickets(STUDENT), memory_events=st.memory_events)

# 2 ---------------------------------------------------------------- consent
mem.grant_consent(STUDENT)
tools.TICKET_STORE.clear()
st, out = session(mem, STUDENT, "Check CASE-1045 and create a support ticket for it.", create=True)
show("2. CONSENT GIVEN: ticket created and a minimal record is saved",
     ticket=out["ticket"]["ticket"]["ticket_id"], stored_records=mem.recall_tickets(STUDENT),
     memory_events=st.memory_events, state=st.snapshot())

# 3 ---------------------------------------------------------------- new session, memory helps
clock.now += timedelta(days=2)
st, out = session(mem, STUDENT, "What is happening with CASE-1045?")
show("3. NEW SESSION 2 DAYS LATER: memory adds an advisory note to the lookup",
     advisory=out["lookup"].get("memory"), memory_events=st.memory_events)

# 4 ---------------------------------------------------------------- duplicate warning
st, out = session(mem, STUDENT, "Check CASE-1045 and create a support ticket for it.", create=True)
show("4. DUPLICATE REQUEST: memory raises a visible warning, no silent block, no silent create",
     ticket_result=out["ticket"], phase_path=[t["to"] for t in st.transitions],
     tickets_in_store=len(tools.TICKET_STORE))

# 5 ---------------------------------------------------------------- student override
st, out = session(mem, STUDENT, "Check CASE-1045 and create another support ticket anyway.", create=True)
show("5. STUDENT OVERRIDES: the student, not memory, makes the final call",
     ticket=out["ticket"].get("ticket", {}).get("ticket_id"), tickets_in_store=len(tools.TICKET_STORE))

# 6 ---------------------------------------------------------------- isolation
st = SessionState(student_id=OTHER); st.memory_enabled = mem.has_consent(OTHER)
res = agent.execute_tool("get_case_status", {"case_id": "CASE-1045"}, st, "status?", mem)
show("6. ANOTHER STUDENT cannot see this case or this memory",
     lookup=res, other_students_memory=mem.recall_tickets(OTHER))

# 7 ---------------------------------------------------------------- export, forget, audit
show("7a. RIGHT OF ACCESS: export", export=mem.export_all(STUDENT))
show("7b. RIGHT TO DELETE: forget", records_deleted=mem.forget(STUDENT), after=mem.recall_tickets(STUDENT))
show("7c. AUDIT TRAIL (no content, only who/what/when)", audit=mem.audit_trail(STUDENT))

# 8 ---------------------------------------------------------------- retention
mem.remember_ticket(STUDENT, {"ticket_id": "TICKET-9001", "case_id": "CASE-1045",
                              "category": "registration"}, "in_progress")
clock.now += timedelta(days=91)
show("8. RETENTION: after 91 days the record is invisible and purged",
     visible=mem.recall_tickets(STUDENT), purged=mem.purge_expired())

# 9 ---------------------------------------------------------------- MCP-style interface
srv = CaseSupportServer(mem)
caller = Caller(STUDENT, {"cases:read", "tickets:write", "memory:read", "memory:delete"})
rpc = lambda m, p=None, c=caller: srv.handle({"jsonrpc": "2.0", "id": 1, "method": m, "params": p or {}}, c)
show("9a. MCP tools/list (full scopes)", tools=[t["name"] for t in rpc("tools/list")["result"]["tools"]])
show("9b. MCP tools/list (read-only caller)",
     tools=[t["name"] for t in rpc("tools/list", c=Caller(STUDENT, {"cases:read"}))["result"]["tools"]])
show("9c. MCP forbidden call (no tickets:write scope)",
     response=rpc("tools/call", {"name": "ticket.create", "arguments": {
         "case_id": "CASE-1045", "category": "registration", "description": "portal issue",
         "confirmed_by_student": True}}, Caller(STUDENT, {"cases:read"})))
show("9d. MCP invalid arguments", response=rpc("tools/call", {"name": "case.get_status", "arguments": {"case_id": "1045"}}))
show("9e. MCP success (student_name is withheld)",
     response=rpc("tools/call", {"name": "case.get_status", "arguments": {"case_id": "CASE-1045"}}))

out_path = Path(__file__).resolve().parents[1] / "docs" / "evidence" / "week6_demo_output.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
print(f"\nEvidence written to {out_path}")
