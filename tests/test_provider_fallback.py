"""Offline tests for the Groq (primary) -> Gemini (fallback) provider layer.

No API keys or network needed: providers are replaced with fakes. These tests
check the application's behaviour around the models (fallback, shared history,
safe failure, argument filtering), not the quality of any real model.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import test_bounded_agent  # noqa: F401  (installs the google-genai stub if the SDK is absent)
import bounded_agent as agent
import tools as project_tools

TICKET_MSG = "Check CASE-1045 and please create a follow-up support ticket for my registration issue."
TICKET_ARGS = {"case_id": "CASE-1045", "category": "registration",
               "description": "Course registration is not reflecting on the portal."}


class FakeProvider:
    """Scripted provider. Each script item is a ModelResponse or an Exception to raise."""

    def __init__(self, name, script):
        self.name = name
        self.script = list(script)
        self.seen_histories = []

    def generate(self, history):
        self.seen_histories.append(json.loads(json.dumps(history, default=str)))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def call(name, args, id=None):
    return agent.ModelResponse(text="", tool_calls=[agent.ToolRequest(name, args, id)])


def say(text):
    return agent.ModelResponse(text=text, tool_calls=[])


def run(providers, message=TICKET_MSG):
    with patch("builtins.print"):
        return agent.run_bounded_agent(message, providers=providers)


class FallbackBehaviourTests(unittest.TestCase):
    def setUp(self):
        project_tools.TICKET_STORE.clear()

    def test_primary_answers_and_fallback_is_never_called(self):
        groq = FakeProvider("groq", [call("get_case_status", {"case_id": "CASE-1045"}),
                                     say("Your case is in progress.")])
        gemini = FakeProvider("gemini", [])
        result = run([groq, gemini])
        self.assertEqual(result["response"], "Your case is in progress.")
        self.assertEqual(gemini.seen_histories, [])
        self.assertTrue(all(e.get("provider") == "groq" for e in result["trace"] if "provider" in e))

    def test_falls_back_when_primary_fails(self):
        groq = FakeProvider("groq", [RuntimeError("503 unavailable")])
        gemini = FakeProvider("gemini", [say("Fallback answer.")])
        result = run([groq, gemini], "Hello")
        self.assertEqual(result["response"], "Fallback answer.")
        failures = [e for e in result["trace"] if e["event"] == "provider_failure"]
        self.assertEqual([f["provider"] for f in failures], ["groq"])
        self.assertEqual(failures[0]["error_type"], "RuntimeError")
        stop = result["trace"][-1]
        self.assertEqual(stop["provider"], "gemini")
        self.assertIn("latency_ms", stop)

    def test_failover_inside_a_round_does_not_use_an_extra_iteration(self):
        groq = FakeProvider("groq", [RuntimeError("down")])
        gemini = FakeProvider("gemini", [say("ok")])
        result = run([groq, gemini], "Hello")
        self.assertEqual(result["state"]["iteration"], 1)

    def test_missing_primary_key_counts_as_primary_failure(self):
        gemini = FakeProvider("gemini", [say("ok")])
        with patch.dict("os.environ", {"GROQ_API_KEY": ""}):
            primary = agent.GroqProvider()
            with self.assertRaises(agent.ProviderError):
                primary.generate([{"type": "user", "text": "hi"}])
            result = run([primary, gemini], "hi")
        failure = [e for e in result["trace"] if e["event"] == "provider_failure"][0]
        self.assertEqual(failure["error_type"], "ProviderError")
        self.assertIn("GROQ_API_KEY", failure["detail"])

    def test_both_fail_stops_safely_without_acting(self):
        groq = FakeProvider("groq", [RuntimeError("down")])
        gemini = FakeProvider("gemini", [RuntimeError("down too")])
        result = run([groq, gemini])
        self.assertEqual(result["response"], agent.SAFE_FAILURE_MESSAGE)
        self.assertEqual(result["state"]["phase"], "failed")
        self.assertEqual(project_tools.TICKET_STORE, {})
        self.assertEqual(result["trace"][-1]["reason"], "All model providers failed.")

    def test_error_details_from_sdk_exceptions_are_not_logged(self):
        groq = FakeProvider("groq", [RuntimeError("secret-token-123 in error text")])
        gemini = FakeProvider("gemini", [say("ok")])
        result = run([groq, gemini], "hi")
        self.assertNotIn("secret-token-123", json.dumps(result["trace"]))

    def test_observability_trace_is_useful_and_excludes_student_content(self):
        provider = FakeProvider("groq", [
            call("get_case_status", {"case_id": "CASE-1045"}),
            say("Private response for student STU-2026-001."),
        ])
        provider.model = "fake-model"
        result = run([provider], "Private student message about CASE-1045.")
        trace = result["observability"]
        encoded = json.dumps(trace)

        self.assertTrue(trace["prompt_version"].startswith("sha256:"))
        self.assertEqual(trace["providers"], [
            {"provider": "groq", "model": "fake-model"},
        ])
        self.assertEqual(trace["retrieval"]["case_status_lookups"], 1)
        self.assertFalse(trace["retrieval"]["document_retrieval_used"])
        self.assertEqual(trace["outcome"]["phase"], "responded")
        self.assertTrue(any(event["event"] == "model_call" for event in trace["events"]))
        tool_event = next(event for event in trace["events"] if event["event"] == "tool_call")
        self.assertEqual(tool_event["input_fields"], ["case_id"])
        self.assertTrue(tool_event["success"])
        self.assertNotIn("CASE-1045", encoded)
        self.assertNotIn("STU-2026-001", encoded)
        self.assertNotIn("Private student message", encoded)
        self.assertNotIn("Private response", encoded)

    def test_tool_exception_is_attached_as_a_redacted_failure_trace(self):
        provider = FakeProvider("groq", [
            call("get_case_status", {"case_id": "CASE-1045"}),
        ])
        with patch("bounded_agent.get_case_status", side_effect=RuntimeError(
            "private case details"
        )):
            with patch("builtins.print"):
                with self.assertRaises(RuntimeError) as raised:
                    agent.run_bounded_agent("Check CASE-1045.", providers=[provider])

        trace = raised.exception.observability
        encoded = json.dumps(trace)
        tool_event = next(event for event in trace["events"] if event["event"] == "tool_call")
        self.assertEqual(trace["outcome"]["phase"], "crashed")
        self.assertEqual(tool_event["error_type"], "RuntimeError")
        self.assertFalse(tool_event["success"])
        self.assertNotIn("CASE-1045", encoded)
        self.assertNotIn("private case details", encoded)

    def test_run_stays_on_fallback_after_switching(self):
        groq = FakeProvider("groq", [RuntimeError("down")])
        gemini = FakeProvider("gemini", [call("get_case_status", {"case_id": "CASE-1045"}),
                                         say("done")])
        result = run([groq, gemini])
        self.assertEqual(len(groq.seen_histories), 1)       # not retried in round 2
        self.assertEqual(len(gemini.seen_histories), 2)

    def test_fallback_sees_earlier_tool_results_and_cannot_repeat_the_ticket(self):
        groq = FakeProvider("groq", [
            call("get_case_status", {"case_id": "CASE-1045"}, "g1"),
            call("create_support_ticket", TICKET_ARGS, "g2"),
            RuntimeError("groq died mid-run"),
        ])
        gemini = FakeProvider("gemini", [
            call("create_support_ticket", TICKET_ARGS),     # tries to create a second ticket
            say("A ticket was already created."),
        ])
        result = run([groq, gemini])

        # The fallback was handed both earlier tool calls and their results.
        first_history = gemini.seen_histories[0]
        self.assertEqual([e["type"] for e in first_history],
                         ["user", "tool_call", "tool_result", "tool_call", "tool_result"])
        # The application, not the prompt, refused the second ticket.
        tool_events = [e for e in result["trace"] if e["event"] == "tool_call"]
        self.assertTrue(tool_events[1]["result"]["success"])
        self.assertFalse(tool_events[2]["result"]["success"])
        self.assertEqual(len(project_tools.TICKET_STORE), 1)


class ArgumentFilteringTests(unittest.TestCase):
    def setUp(self):
        project_tools.TICKET_STORE.clear()

    def test_unexpected_arguments_are_dropped_and_traced(self):
        groq = FakeProvider("groq", [
            call("get_case_status", {"case_id": "CASE-1045", "student_id": "STU-2026-002"}),
            say("ok"),
        ])
        result = run([groq], "Check CASE-1045")
        event = [e for e in result["trace"] if e["event"] == "tool_call"][0]
        self.assertEqual(event["arguments"], {"case_id": "CASE-1045"})
        self.assertEqual(event["dropped_arguments"], ["student_id"])

    def test_unknown_tool_is_still_refused(self):
        groq = FakeProvider("groq", [call("delete_all_cases", {"x": 1}), say("sorry")])
        result = run([groq], "Delete everything")
        event = [e for e in result["trace"] if e["event"] == "tool_call"][0]
        self.assertFalse(event["result"]["success"])

    def test_multiple_tool_requests_still_rejected(self):
        two = agent.ModelResponse(text="", tool_calls=[
            agent.ToolRequest("get_case_status", {"case_id": "CASE-1045"}, "a"),
            agent.ToolRequest("get_case_status", {"case_id": "CASE-2031"}, "b"),
        ])
        result = run([FakeProvider("groq", [two])], "Check both")
        self.assertEqual(result["state"]["phase"], "failed")


class HistoryConversionTests(unittest.TestCase):
    HISTORY = [
        {"type": "user", "text": "Check CASE-1045"},
        {"type": "tool_call", "id": "g1", "name": "get_case_status",
         "args": {"case_id": "CASE-1045"}, "provider": "groq", "native": None},
        {"type": "tool_result", "id": "g1", "name": "get_case_status",
         "result": {"success": True, "status": "in_progress"}},
    ]

    def test_groq_message_shapes(self):
        msgs = agent.history_to_groq_messages(self.HISTORY)
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "tool"])
        call_ = msgs[1]["tool_calls"][0]
        self.assertEqual(call_["id"], "g1")
        self.assertEqual(json.loads(call_["function"]["arguments"]), {"case_id": "CASE-1045"})
        self.assertEqual(msgs[2]["tool_call_id"], "g1")
        self.assertEqual(json.loads(msgs[2]["content"])["status"], "in_progress")

    def test_gemini_gets_other_models_tool_calls_as_plain_text(self):
        contents = agent.history_to_gemini_contents(self.HISTORY)
        self.assertEqual(len(contents), 1)
        self.assertIsInstance(contents[0], str)
        self.assertIn("Check CASE-1045", contents[0])
        self.assertIn("[Earlier tool call", contents[0])
        self.assertIn("in_progress", contents[0])

    def test_gemini_replays_its_own_tool_calls_natively(self):
        history = [
            {"type": "user", "text": "Check CASE-1045"},
            {"type": "tool_call", "id": "call_1", "name": "get_case_status",
             "args": {"case_id": "CASE-1045"}, "provider": "gemini", "native": "NATIVE-CONTENT"},
            {"type": "tool_result", "id": "call_1", "name": "get_case_status",
             "result": {"success": True}},
        ]
        contents = agent.history_to_gemini_contents(history)
        self.assertEqual(contents[0], "Check CASE-1045")
        self.assertEqual(contents[1], "NATIVE-CONTENT")
        self.assertEqual(len(contents), 3)


class GroqResponseParsingTests(unittest.TestCase):
    """GroqProvider against a fake client shaped like Groq's chat-completions response."""

    @staticmethod
    def fake_client(message):
        from types import SimpleNamespace as NS
        create = lambda **kwargs: NS(choices=[NS(message=message)])
        client = NS(chat=NS(completions=NS(create=create)))
        client.last_kwargs = None
        return client

    def test_tool_call_is_parsed(self):
        from types import SimpleNamespace as NS
        msg = NS(content=None, tool_calls=[NS(id="abc", function=NS(
            name="get_case_status", arguments='{"case_id": "CASE-1045"}'))])
        out = agent.GroqProvider(client=self.fake_client(msg)).generate([{"type": "user", "text": "hi"}])
        self.assertEqual(out.tool_calls[0].name, "get_case_status")
        self.assertEqual(out.tool_calls[0].args, {"case_id": "CASE-1045"})
        self.assertEqual(out.tool_calls[0].id, "abc")

    def test_invalid_json_arguments_become_empty_and_are_rejected_by_the_app(self):
        from types import SimpleNamespace as NS
        msg = NS(content=None, tool_calls=[NS(id="x", function=NS(
            name="get_case_status", arguments="{not json"))])
        out = agent.GroqProvider(client=self.fake_client(msg)).generate([{"type": "user", "text": "hi"}])
        self.assertEqual(out.tool_calls[0].args, {})
        result = run([FakeProvider("groq", [out, say("please give me the case ID")])], "Check my case")
        event = [e for e in result["trace"] if e["event"] == "tool_call"][0]
        self.assertFalse(event["result"]["success"])

    def test_plain_answer_with_no_content_is_empty_text(self):
        from types import SimpleNamespace as NS
        msg = NS(content=None, tool_calls=None)
        out = agent.GroqProvider(client=self.fake_client(msg)).generate([{"type": "user", "text": "hi"}])
        self.assertEqual((out.text, out.tool_calls), ("", []))


class MissingKeyTests(unittest.TestCase):
    def test_missing_gemini_key_does_not_stop_the_module_loading(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}):
            self.assertIsNone(agent._make_gemini_client())

    def test_gemini_without_a_client_is_a_provider_failure(self):
        with patch.object(agent, "client", None):
            with self.assertRaises(agent.ProviderError):
                agent.GeminiProvider().generate([{"type": "user", "text": "hi"}])

    def test_groq_works_when_gemini_has_no_key(self):
        groq = FakeProvider("groq", [say("Groq answered.")])
        with patch.object(agent, "client", None):
            result = run([groq, agent.GeminiProvider()], "hi")
        self.assertEqual(result["response"], "Groq answered.")


if __name__ == "__main__":
    unittest.main()
