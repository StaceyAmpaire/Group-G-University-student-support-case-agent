"""
Offline tests for agent_loop.py. No network or API keys needed: the models are
replaced by scripted fakes, so these tests check the loop's own rules.

Run from src/:   python -m unittest test_agent_loop -v
"""

import copy
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import agent_loop
import tools
from agent_loop import ModelResponse, ToolRequest, _case_ids_in, _run_turn
from memory import MemoryStore

PEPPER = "test-pepper-0123456789"

logging.getLogger("agent_loop").setLevel(logging.CRITICAL)  # keep expected failures quiet


def call(name, **arguments):
    return ToolRequest(id=f"call_{name}", name=name, arguments=arguments)


def say(text):
    return ModelResponse(text=text, tool_requests=[])


def use(*requests):
    return ModelResponse(text="", tool_requests=list(requests))


class Scripted:
    """A fake provider that returns scripted responses (or raises scripted errors)."""

    def __init__(self, name, script):
        self.name = name
        self.script = list(script)
        self.seen_histories = []

    def generate(self, history):
        self.seen_histories.append(copy.deepcopy(history))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class LoopTests(unittest.TestCase):
    def setUp(self):
        tools.TICKET_STORE.clear()
        self.conv = f"conv-{self.id()}"
        agent_loop.reset_conversation(self.conv)

    def turn(self, message, providers, student_id=None):
        return _run_turn(message, self.conv, student_id, providers)

    # --- the three contract traces ---------------------------------------

    def test_trace1_status_lookup(self):
        model = Scripted("groq", [use(call("get_case_status", case_id="CASE-2031")), say("It is resolved.")])
        result = self.turn("status of CASE-2031?", [model])
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(result["outcome"], "goal_reached")
        self.assertEqual(result["tool_trace"][0]["result"]["status"], "resolved")
        self.assertEqual(result["reply"], "It is resolved.")

    def test_trace2_ticket_created_within_three_rounds(self):
        model = Scripted("groq", [
            use(call("get_case_status", case_id="CASE-1045")),
            use(call("create_support_ticket", case_id="CASE-1045", category="registration",
                     description="Course units missing")),
            say("Ticket TICKET-1001 created."),
        ])
        result = self.turn("open a ticket for CASE-1045", [model])
        self.assertEqual(result["rounds"], 3)
        self.assertEqual(result["outcome"], "goal_reached")
        self.assertEqual(len(tools.TICKET_STORE), 1)
        state = agent_loop._STATES[self.conv]
        self.assertTrue(state.ticket_created)
        self.assertIn(state.ticket_id, tools.TICKET_STORE)

    def test_trace3_unknown_case_creates_no_ticket(self):
        model = Scripted("groq", [
            use(call("create_support_ticket", case_id="CASE-9999", category="fees", description="Fee issue")),
            say("I could not find that case. Please check the ID."),
        ])
        result = self.turn("ticket for CASE-9999", [model])
        self.assertEqual(len(tools.TICKET_STORE), 0)
        self.assertFalse(agent_loop._STATES[self.conv].ticket_created)
        self.assertEqual(result["outcome"], "needs_student_input")
        sources = [e["source"] for e in result["tool_trace"]]
        self.assertEqual(sources, ["application", "model"])  # app lookup, then the rejected ticket call
        self.assertFalse(result["tool_trace"][1]["result"]["success"])

    # --- limits and guards -----------------------------------------------

    def test_one_ticket_per_conversation(self):
        ticket = call("create_support_ticket", case_id="CASE-1045", category="registration", description="x")
        model = Scripted("groq", [use(ticket), say("Done."), use(ticket), say("Already have one.")])
        self.turn("first ticket for CASE-1045", [model])
        second = self.turn("second ticket for CASE-1045", [model])
        self.assertEqual(len(tools.TICKET_STORE), 1)
        blocked = [e for e in second["tool_trace"] if e["tool"] == "create_support_ticket"][0]
        self.assertTrue(blocked["result"]["blocked"])

    def test_escalated_case_gets_no_ticket(self):
        model = Scripted("groq", [
            use(call("create_support_ticket", case_id="CASE-3312", category="it_support", description="x")),
            say("It is escalated; a human team must handle it."),
        ])
        result = self.turn("ticket for CASE-3312", [model])
        self.assertEqual(len(tools.TICKET_STORE), 0)
        self.assertEqual(result["outcome"], "handed_off")

    def test_resolved_case_gets_no_ticket(self):
        model = Scripted("groq", [
            use(call("create_support_ticket", case_id="CASE-2031", category="fees", description="x")),
            say("That case is resolved."),
        ])
        result = self.turn("ticket for CASE-2031", [model])
        self.assertEqual(len(tools.TICKET_STORE), 0)
        self.assertEqual(result["outcome"], "goal_reached")

    def test_missing_argument_does_not_call_tool(self):
        model = Scripted("groq", [
            use(call("create_support_ticket", case_id="CASE-1045")),  # category, description absent
            say("What category and description should I use?"),
        ])
        result = self.turn("open a ticket", [model])  # must not raise TypeError
        entry = result["tool_trace"][0]
        self.assertFalse(entry["result"]["success"])
        self.assertIn("category", entry["result"]["error"])
        self.assertEqual(len(tools.TICKET_STORE), 0)
        self.assertTrue(result["clarification_asked"])

    def test_unknown_tool_is_refused(self):
        model = Scripted("groq", [use(call("escalate_case", case_id="CASE-1045")), say("I cannot escalate.")])
        result = self.turn("escalate my case", [model])
        self.assertFalse(result["tool_trace"][0]["result"]["success"])
        self.assertIn("not permitted", result["tool_trace"][0]["result"]["error"])

    def test_model_supplied_student_id_is_ignored(self):
        model = Scripted("groq", [
            use(call("get_case_status", case_id="CASE-1045", student_id="someone-else")),
            say("In progress."),
        ])
        result = self.turn("status", [model], student_id="stu-1")
        entry = result["tool_trace"][0]
        self.assertNotIn("student_id", entry["arguments"])
        self.assertEqual(entry["ignored_arguments"], ["student_id"])

    def test_round_limit_stops_without_running_extra_tools(self):
        lookup = use(call("get_case_status", case_id="CASE-1045"))
        model = Scripted("groq", [lookup, lookup, lookup])
        result = self.turn("loop forever", [model])
        self.assertEqual(result["rounds"], agent_loop.MAX_TOOL_ROUNDS)
        self.assertEqual(result["outcome"], "round_limit")
        self.assertEqual(result["reply"], agent_loop.SAFE_ROUND_LIMIT_MESSAGE)
        self.assertEqual(len(result["tool_trace"]), agent_loop.MAX_TOOL_ROUNDS - 1)

    # --- fallback ----------------------------------------------------------

    def test_fallback_receives_tool_history_and_cannot_repeat_a_ticket(self):
        ticket = call("create_support_ticket", case_id="CASE-1045", category="registration", description="x")
        primary = Scripted("groq", [use(ticket), say("Ticket created."), RuntimeError("groq down")])
        fallback = Scripted("gemini", [use(ticket), say("A ticket already exists.")])
        providers = [primary, fallback]

        self.turn("open a ticket for CASE-1045", providers)
        second = self.turn("open another ticket for CASE-1045", providers)

        self.assertTrue(second["fallback_used"])
        self.assertEqual(second["model_used"], "gemini")
        self.assertIn("groq: RuntimeError", second["model_errors"][0])
        roles = [m["role"] for m in fallback.seen_histories[0]]
        self.assertIn("tool", roles)  # the earlier tool call and result reached the fallback
        self.assertEqual(len(tools.TICKET_STORE), 1)

    def test_fallback_stays_active_for_rest_of_turn(self):
        primary = Scripted("groq", [RuntimeError("down")])
        fallback = Scripted("gemini", [use(call("get_case_status", case_id="CASE-1045")), say("In progress.")])
        result = self.turn("status", [primary, fallback])
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(len(primary.seen_histories), 1)  # primary not retried within the turn

    def test_both_models_failing_gives_safe_message(self):
        result = self.turn("hi", [Scripted("groq", [RuntimeError("a")]), Scripted("gemini", [RuntimeError("b")])])
        self.assertEqual(result["outcome"], "unrecoverable_failure")
        self.assertEqual(result["reply"], agent_loop.SAFE_FAILURE_MESSAGE)
        self.assertEqual(result["tool_trace"], [])

    def test_empty_model_response_triggers_fallback(self):
        result = self.turn("hi", [Scripted("groq", [say("   ")]), Scripted("gemini", [say("Hello.")])])
        self.assertEqual(result["model_used"], "gemini")

    def test_tool_crash_is_unrecoverable(self):
        original = tools.TOOL_FUNCTIONS["get_case_status"]
        tools.TOOL_FUNCTIONS["get_case_status"] = lambda case_id: 1 / 0
        try:
            model = Scripted("groq", [use(call("get_case_status", case_id="CASE-1045"))])
            with self.assertLogs("agent_loop", level="ERROR"):
                result = self.turn("status", [model])
        finally:
            tools.TOOL_FUNCTIONS["get_case_status"] = original
        self.assertEqual(result["outcome"], "unrecoverable_failure")
        self.assertEqual(result["reply"], agent_loop.SAFE_FAILURE_MESSAGE)

    # --- conversation state --------------------------------------------------

    def test_conversation_cannot_switch_student(self):
        model = Scripted("groq", [say("Hi."), say("Hi again.")])
        self.turn("hello", [model], student_id="stu-1")
        with self.assertRaises(PermissionError):
            self.turn("hello", [model], student_id="stu-2")

    def test_blank_inputs_rejected(self):
        with self.assertRaises(ValueError):
            _run_turn("  ", self.conv, None, [])
        with self.assertRaises(ValueError):
            _run_turn("hello", "", None, [])


class SpyMemory:
    """Records calls so tests can prove memory was (or was not) touched."""

    def __init__(self):
        self.calls = []

    def recall(self, student_id):
        self.calls.append(("recall", student_id))
        return None

    def remember_case(self, student_id, case_id):
        self.calls.append(("remember_case", student_id, case_id))

    def remember_ticket(self, student_id, ticket, case_status=None):
        self.calls.append(("remember_ticket", student_id, ticket["ticket_id"], case_status))

    def find_recent_ticket(self, student_id, case_id):
        self.calls.append(("find_recent_ticket", student_id, case_id))
        return None


class BrokenMemory:
    def recall(self, student_id):
        raise RuntimeError("disk error")

    def remember_case(self, *args):
        raise RuntimeError("disk error")

    def remember_ticket(self, *args):
        raise RuntimeError("disk error")

    def find_recent_ticket(self, *args):
        raise RuntimeError("disk error")


def notes_in(history):
    return [m for m in history if m["role"] == "user" and m["text"].startswith("[Application note]")]


class MemoryIntegrationTests(unittest.TestCase):
    def setUp(self):
        tools.TICKET_STORE.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "memory.db"
        self.memory = MemoryStore(self.path, pepper=PEPPER)
        self.memory.grant_consent("stu-1")  # the application records the student's opt-in
        self.conv = f"conv-{self.id()}"
        agent_loop.reset_conversation(self.conv)
        agent_loop.reset_conversation(self.conv + "-b")

    def turn(self, message, providers, student_id="stu-1", conv=None, memory="store"):
        memory = self.memory if memory == "store" else memory
        return _run_turn(message, conv or self.conv, student_id, providers, memory=memory)

    # --- saving ------------------------------------------------------------

    def test_successful_lookup_is_saved(self):
        model = Scripted("groq", [use(call("get_case_status", case_id="CASE-2031")), say("Resolved.")])
        self.turn("status of CASE-2031", [model])
        self.assertEqual(self.memory.recall("stu-1")["active_case_id"], "CASE-2031")

    def test_not_found_lookup_is_not_saved(self):
        model = Scripted("groq", [use(call("get_case_status", case_id="CASE-9999")), say("Not found.")])
        self.turn("status of CASE-9999", [model])
        self.assertIsNone(self.memory.recall("stu-1"))

    def test_ticket_is_saved(self):
        model = Scripted("groq", [
            use(call("create_support_ticket", case_id="CASE-1045", category="registration", description="x")),
            say("Ticket opened."),
        ])
        self.turn("open a ticket for CASE-1045", [model])
        record = self.memory.recall("stu-1")
        self.assertTrue(record["tickets"][0]["ticket_id"].startswith("TICKET-"))
        self.assertEqual(record["tickets"][0]["category"], "registration")
        self.assertEqual(record["tickets"][0]["case_status"], "in_progress")  # status when it was opened
        self.assertEqual(record["active_case_id"], "CASE-1045")

    def test_nothing_is_saved_or_read_without_a_student_id(self):
        spy = SpyMemory()
        model = Scripted("groq", [use(call("get_case_status", case_id="CASE-2031")), say("Resolved.")])
        self.turn("status of CASE-2031", [model], student_id=None, memory=spy)
        self.assertEqual(spy.calls, [])

    def test_memory_only_stores_what_the_policy_allows(self):
        model = Scripted("groq", [use(call("get_case_status", case_id="CASE-1045")), say("In progress.")])
        self.turn("status of CASE-1045. My name is Grace and my number is 0700000000", [model])
        record = self.memory.recall("stu-1")
        self.assertEqual(record["tickets"], [])
        self.assertEqual(record["active_case_id"], "CASE-1045")
        import sqlite3
        conn = sqlite3.connect(self.path)
        try:
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            text = " ".join(str(v) for t in tables for row in conn.execute(f"SELECT * FROM {t}") for v in row)
        finally:
            conn.close()
        self.assertNotIn("Grace", text)
        self.assertNotIn("0700000000", text)

    # --- using memory in a later conversation --------------------------------

    def test_note_reaches_the_model_once_and_survives_a_restart(self):
        self.memory.remember_case("stu-1", "CASE-1045")
        restarted = MemoryStore(self.path, pepper=PEPPER)  # a new process would open the same file
        model = Scripted("groq", [say("Hello."), say("Hello again.")])
        self.turn("hi", [model], memory=restarted)
        self.turn("hi again", [model], memory=restarted)
        first_note = notes_in(model.seen_histories[0])
        self.assertEqual(len(first_note), 1)
        self.assertIn("CASE-1045", first_note[0]["text"])
        self.assertEqual(len(notes_in(model.seen_histories[1])), 1)  # not repeated

    def test_no_note_when_nothing_is_remembered(self):
        model = Scripted("groq", [say("Hello.")])
        self.turn("hi", [model])
        self.assertEqual(notes_in(model.seen_histories[0]), [])

    def test_remembered_case_can_be_looked_up_read_only(self):
        self.memory.remember_case("stu-1", "CASE-1045")
        model = Scripted("groq", [use(call("get_case_status", case_id="CASE-1045")), say("It is in progress.")])
        result = self.turn("what is the status of my case?", [model])
        self.assertTrue(result["tool_trace"][0]["result"]["success"])
        self.assertEqual(result["tool_trace"][0]["result"]["status"], "in_progress")  # fresh, not from memory

    def test_remembered_case_cannot_open_a_ticket_until_the_student_gives_the_id(self):
        self.memory.remember_case("stu-1", "CASE-1045")
        ticket = call("create_support_ticket", case_id="CASE-1045", category="registration", description="x")
        model = Scripted("groq", [
            use(ticket), say("Please confirm the case ID."),
            use(ticket), say("Ticket opened."),
        ])
        first = self.turn("please open a ticket for my case", [model])
        self.assertEqual(len(tools.TICKET_STORE), 0)
        self.assertIn("has not given", first["tool_trace"][0]["result"]["error"])
        self.assertEqual(first["outcome"], "needs_student_input")

        second = self.turn("It is CASE-1045", [model])
        self.assertEqual(len(tools.TICKET_STORE), 1)
        self.assertEqual(second["outcome"], "goal_reached")

    def test_forget_removes_what_the_next_conversation_sees(self):
        self.memory.remember_case("stu-1", "CASE-1045")
        self.assertTrue(self.memory.forget("stu-1"))
        model = Scripted("groq", [say("Hello.")])
        self.turn("hi", [model], conv=self.conv + "-b")
        self.assertEqual(notes_in(model.seen_histories[0]), [])

    # --- consent -------------------------------------------------------------------

    def test_without_consent_nothing_is_saved_and_no_note_is_given(self):
        model = Scripted("groq", [use(call("get_case_status", case_id="CASE-2031")), say("Resolved.")])
        self.turn("status of CASE-2031", [model], student_id="stu-no-consent")
        self.assertEqual(notes_in(model.seen_histories[0]), [])
        self.assertIsNone(self.memory.recall("stu-no-consent"))
        self.assertEqual(self.memory.export_all("stu-no-consent")["records"], [])

    # --- duplicate-ticket warning ------------------------------------------------------

    def remembered_ticket(self):
        self.memory.remember_ticket(
            "stu-1", {"ticket_id": "TICKET-1001", "case_id": "CASE-1045", "category": "registration"}, "in_progress"
        )
        return call("create_support_ticket", case_id="CASE-1045", category="registration", description="x")

    def test_duplicate_is_held_back_once_then_the_student_can_proceed(self):
        ticket = self.remembered_ticket()
        model = Scripted("groq", [
            use(ticket), say("You opened TICKET-1001 on CASE-1045 earlier. Do you want another?"),
            use(ticket), say("Done, I've opened another ticket."),
        ])
        first = self.turn("please open a ticket for CASE-1045", [model])
        self.assertEqual(len(tools.TICKET_STORE), 0)
        self.assertIn("TICKET-1001", first["tool_trace"][-1]["result"]["error"])
        self.assertEqual(first["outcome"], "needs_student_input")

        second = self.turn("Yes, please open another one for CASE-1045", [model])
        self.assertEqual(len(tools.TICKET_STORE), 1)  # memory advised; the student decided
        self.assertEqual(second["outcome"], "goal_reached")

    def test_duplicate_retry_in_the_same_turn_stays_blocked(self):
        ticket = self.remembered_ticket()
        model = Scripted("groq", [use(ticket), use(ticket), say("Please confirm you want another ticket.")])
        self.turn("open a ticket for CASE-1045", [model])
        self.assertEqual(len(tools.TICKET_STORE), 0)

    def test_a_ticket_on_a_different_case_is_not_held_back(self):
        self.memory.remember_ticket(
            "stu-1", {"ticket_id": "TICKET-1001", "case_id": "CASE-2031", "category": "fees"}, "resolved"
        )
        ticket = call("create_support_ticket", case_id="CASE-1045", category="registration", description="x")
        model = Scripted("groq", [use(ticket), say("Opened.")])
        self.turn("open a ticket for CASE-1045", [model])
        self.assertEqual(len(tools.TICKET_STORE), 1)

    # --- failure isolation ------------------------------------------------------

    def test_broken_memory_does_not_break_a_turn(self):
        ticket = call("create_support_ticket", case_id="CASE-1045", category="registration", description="x")
        model = Scripted("groq", [use(ticket), say("Ticket opened.")])
        result = self.turn("open a ticket for CASE-1045", [model], memory=BrokenMemory())
        self.assertEqual(result["outcome"], "goal_reached")
        self.assertEqual(len(tools.TICKET_STORE), 1)


class CaseReferenceTests(unittest.TestCase):
    def test_references_are_normalised(self):
        self.assertEqual(_case_ids_in("case 1045, CASE-2031 and case#3312"),
                         {"CASE-1045", "CASE-2031", "CASE-3312"})

    def test_words_that_merely_contain_case_do_not_match(self):
        self.assertEqual(_case_ids_in("my briefcase 5 and showcase 12"), set())


class AdapterTests(unittest.TestCase):
    """Check the wire-format conversion and response parsing without a network."""

    HISTORY = [
        {"role": "user", "text": "ticket please"},
        {"role": "assistant", "text": "", "tool_calls": [
            {"id": "c1", "name": "create_support_ticket", "arguments": {"case_id": "CASE-1045"}}]},
        {"role": "tool", "id": "c1", "name": "create_support_ticket", "result": {"success": False}},
        {"role": "assistant", "text": "Need a category."},
        {"role": "user", "text": "registration"},
    ]

    def test_groq_messages(self):
        messages = agent_loop.GroqProvider._messages(self.HISTORY)
        self.assertEqual([m["role"] for m in messages],
                         ["system", "user", "assistant", "tool", "assistant", "user"])
        self.assertEqual(messages[2]["tool_calls"][0]["function"]["name"], "create_support_ticket")
        self.assertIsNone(messages[2]["content"])
        self.assertEqual(messages[3]["tool_call_id"], "c1")

    def test_groq_generate_parses_tool_calls(self):
        function = SimpleNamespace(name="get_case_status", arguments='{"case_id": "CASE-1045"}')
        message = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="c9", function=function)])
        completions = SimpleNamespace(create=lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=message)]))
        provider = agent_loop.GroqProvider()
        provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        response = provider.generate(self.HISTORY)
        self.assertEqual(response.tool_requests[0].arguments, {"case_id": "CASE-1045"})

    def test_groq_bad_json_arguments_become_none(self):
        self.assertIsNone(agent_loop._parse_arguments("{not json"))

    def test_gemini_turns_merge_and_render_tool_history_as_text(self):
        turns = agent_loop.GeminiProvider._turns(self.HISTORY)
        self.assertEqual([role for role, _ in turns], ["user", "model", "user", "model", "user"])
        self.assertTrue(turns[1][1][0].startswith("[Tool call] create_support_ticket"))
        self.assertTrue(turns[2][1][0].startswith("[Tool result] create_support_ticket"))

    def test_gemini_generate_parses_function_call_and_text(self):
        from google.genai import types

        content = types.Content(role="model", parts=[
            types.Part.from_function_call(name="get_case_status", args={"case_id": "CASE-1045"})])
        reply = types.GenerateContentResponse(candidates=[types.Candidate(content=content)])
        provider = agent_loop.GeminiProvider()
        provider._client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: reply))
        response = provider.generate(self.HISTORY)
        self.assertEqual(response.tool_requests[0].name, "get_case_status")
        self.assertEqual(response.tool_requests[0].arguments, {"case_id": "CASE-1045"})


if __name__ == "__main__":
    unittest.main()