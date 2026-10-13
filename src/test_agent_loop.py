"""
Offline tests for agent_loop.py. No network or API keys needed: the models are
replaced by scripted fakes, so these tests check the loop's own rules.

Run from src/:   python -m unittest test_agent_loop -v
"""

import copy
import logging
import unittest
from types import SimpleNamespace

import agent_loop
import tools
from agent_loop import ModelResponse, ToolRequest, _run_turn

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
        self.turn("first", [model])
        second = self.turn("second", [model])
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

        self.turn("open a ticket", providers)
        second = self.turn("open another ticket", providers)

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
