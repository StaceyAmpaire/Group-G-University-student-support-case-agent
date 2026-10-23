"""Offline tests for the bounded agent's application-side controls.

These tests mock model responses. They do not call Gemini or require an API key.
"""
from __future__ import annotations

import sys
import types as pytypes
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def _install_google_genai_test_stub_if_missing():
    """Supply the tiny SDK surface used by this module if google-genai is absent."""
    try:
        import google.genai  # noqa: F401
        return
    except ImportError:
        pass

    google_module = pytypes.ModuleType("google")
    genai_module = pytypes.ModuleType("google.genai")
    sdk_types = pytypes.ModuleType("google.genai.types")

    class SDKObject:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Part:
        @staticmethod
        def from_function_response(name, response):
            return SDKObject(function_response={"name": name, "response": response})

    class Client:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.models = SDKObject(generate_content=None)

    for name in ("Schema", "FunctionDeclaration", "Tool", "GenerateContentConfig", "Content"):
        setattr(sdk_types, name, type(name, (SDKObject,), {}))
    sdk_types.Part = Part
    genai_module.types = sdk_types
    genai_module.Client = Client
    google_module.genai = genai_module
    sys.modules.setdefault("google", google_module)
    sys.modules.setdefault("google.genai", genai_module)
    sys.modules.setdefault("google.genai.types", sdk_types)


_install_google_genai_test_stub_if_missing()
import bounded_agent as agent  # noqa: E402
import tools as project_tools  # noqa: E402


class BoundedAgentAuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "checked_case_id": None,
            "case_result": None,
            "ticket_created": False,
        }
        project_tools.TICKET_STORE.clear()

    def test_follow_up_support_ticket_wording_is_explicit(self):
        self.assertTrue(agent.has_explicit_ticket_request(
            "Please create a follow-up support ticket for this registration issue."
        ))

    def test_common_explicit_ticket_requests_are_recognized(self):
        for message in (
            "Create a support ticket.",
            "Open a ticket for this issue.",
            "Please raise a ticket.",
            "Submit a support ticket for me.",
        ):
            with self.subTest(message=message):
                self.assertTrue(agent.has_explicit_ticket_request(message))

    def test_ticket_request_negation_is_not_authorization(self):
        for message in (
            "Do not create a support ticket.",
            "Don't open a ticket.",
            "I do not want a ticket.",
            "There is no need to create a ticket.",
            "How do I create a support ticket?",
            "Should I open a ticket?",
        ):
            with self.subTest(message=message):
                self.assertFalse(agent.has_explicit_ticket_request(message))

    def test_successful_ticket_creation_after_successful_case_check(self):
        lookup = agent.execute_tool(
            "get_case_status", {"case_id": "case-1045"}, self.state, "Check case-1045"
        )
        self.assertTrue(lookup["success"])
        self.assertEqual(self.state["checked_case_id"], "CASE-1045")

        result = agent.execute_tool(
            "create_support_ticket",
            {
                "case_id": "CASE-1045",
                "category": "registration",
                "description": "Course registration is not reflecting on the portal.",
            },
            self.state,
            "Check CASE-1045. Please create a follow-up support ticket for this issue.",
        )
        self.assertTrue(result["success"], result)
        self.assertTrue(result["ticket"]["ticket_id"].startswith("TICKET-"))
        self.assertTrue(self.state["ticket_created"])
        self.assertIn(result["ticket"]["ticket_id"], project_tools.TICKET_STORE)

    def test_ticket_creation_is_rejected_without_successful_case_check(self):
        result = agent.execute_tool(
            "create_support_ticket",
            {"case_id": "CASE-1045", "category": "registration", "description": "Portal issue"},
            self.state,
            "Create a support ticket.",
        )
        self.assertFalse(result["success"])
        self.assertIn("Check the existing case", result["error"])
        self.assertEqual(project_tools.TICKET_STORE, {})

    def test_ticket_creation_is_rejected_when_case_does_not_exist(self):
        lookup = agent.execute_tool(
            "get_case_status", {"case_id": "CASE-9999"}, self.state, "Check case"
        )
        self.assertFalse(lookup["success"])
        result = agent.execute_tool(
            "create_support_ticket",
            {"case_id": "CASE-9999", "category": "registration", "description": "Portal issue"},
            self.state,
            "Create a support ticket.",
        )
        self.assertFalse(result["success"])
        self.assertEqual(project_tools.TICKET_STORE, {})

    def test_ticket_creation_is_rejected_for_resolved_case(self):
        agent.execute_tool("get_case_status", {"case_id": "CASE-2031"}, self.state, "Check case")
        result = agent.execute_tool(
            "create_support_ticket",
            {"case_id": "CASE-2031", "category": "fees", "description": "PRN issue"},
            self.state,
            "Create a support ticket.",
        )
        self.assertFalse(result["success"])
        self.assertIn("status is in progress or under review", result["error"])
        self.assertEqual(project_tools.TICKET_STORE, {})

    def test_ticket_cannot_be_created_twice_in_same_run(self):
        agent.execute_tool("get_case_status", {"case_id": "CASE-1045"}, self.state, "Check case")
        args = {
            "case_id": "CASE-1045",
            "category": "registration",
            "description": "Portal issue",
        }
        message = "Please create a follow-up support ticket."
        first = agent.execute_tool("create_support_ticket", args, self.state, message)
        second = agent.execute_tool("create_support_ticket", args, self.state, message)
        self.assertTrue(first["success"])
        self.assertFalse(second["success"])
        self.assertIn("already been created", second["error"])
        self.assertEqual(len(project_tools.TICKET_STORE), 1)

    def test_failed_lookup_clears_previous_case_authorization(self):
        agent.execute_tool("get_case_status", {"case_id": "CASE-1045"}, self.state, "Check case")
        failed_lookup = agent.execute_tool(
            "get_case_status", {"case_id": "CASE-9999"}, self.state, "Check another case"
        )
        self.assertFalse(failed_lookup["success"])
        self.assertIsNone(self.state["checked_case_id"])
        result = agent.execute_tool(
            "create_support_ticket",
            {"case_id": "CASE-1045", "category": "registration", "description": "Portal issue"},
            self.state,
            "Create a support ticket.",
        )
        self.assertFalse(result["success"])
        self.assertEqual(project_tools.TICKET_STORE, {})

    def test_null_description_is_rejected_instead_of_stringified(self):
        agent.execute_tool("get_case_status", {"case_id": "CASE-1045"}, self.state, "Check case")
        result = agent.execute_tool(
            "create_support_ticket",
            {"case_id": "CASE-1045", "category": "registration", "description": None},
            self.state,
            "Create a support ticket.",
        )
        self.assertFalse(result["success"])
        self.assertIn("description is required", result["error"])
        self.assertEqual(project_tools.TICKET_STORE, {})

    def test_ticket_creation_is_rejected_for_a_different_case(self):
        agent.execute_tool("get_case_status", {"case_id": "CASE-1045"}, self.state, "Check case")
        result = agent.execute_tool(
            "create_support_ticket",
            {"case_id": "CASE-2031", "category": "fees", "description": "PRN issue"},
            self.state,
            "Create a support ticket.",
        )
        self.assertFalse(result["success"])
        self.assertIn("must match the checked case", result["error"])
        self.assertEqual(project_tools.TICKET_STORE, {})

    def test_invalid_tool_name_is_rejected(self):
        result = agent.execute_tool("delete_university_record", {}, self.state, "Do it")
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "Tool is not authorized.")


class BoundedAgentLoopTests(unittest.TestCase):
    def setUp(self):
        project_tools.TICKET_STORE.clear()

    @staticmethod
    def model_response(function_calls=None, text="", content_marker="model-content"):
        return pytypes.SimpleNamespace(
            function_calls=function_calls or [],
            text=text,
            candidates=[pytypes.SimpleNamespace(content=content_marker)],
        )

    @staticmethod
    def function_call(name, args):
        return pytypes.SimpleNamespace(name=name, args=args)

    def test_end_to_end_case_lookup_then_ticket_creation(self):
        responses = [
            self.model_response([self.function_call("get_case_status", {"case_id": "CASE-1045"})]),
            self.model_response([self.function_call("create_support_ticket", {
                "case_id": "CASE-1045",
                "category": "registration",
                "description": "Course registration is not reflecting on the portal.",
            })]),
            self.model_response(text="The simulated support ticket was created successfully."),
        ]
        fake_client = pytypes.SimpleNamespace(
            models=pytypes.SimpleNamespace(generate_content=Mock(side_effect=responses))
        )
        with patch.object(agent, "client", fake_client), patch("builtins.print"):
            result = agent.run_bounded_agent(
                "Check CASE-1045. Please create a follow-up support ticket for this issue.",
                student_id="STU-2026-001",   # Week 7: unauthenticated requests are refused
                providers=[agent.GeminiProvider()],
            )

        tool_events = [event for event in result["trace"] if event["event"] == "tool_call"]
        self.assertEqual([event["tool"] for event in tool_events], [
            "get_case_status", "create_support_ticket"
        ])
        self.assertTrue(tool_events[1]["result"]["success"])
        self.assertTrue(result["state"]["ticket_created"])
        self.assertEqual(fake_client.models.generate_content.call_count, 3)
        self.assertEqual(result["response"], "The simulated support ticket was created successfully.")

    def test_iteration_limit_stops_after_maximum_number_of_tool_calls(self):
        repeating_response = self.model_response([
            self.function_call("get_case_status", {"case_id": "CASE-1045"})
        ])
        fake_client = pytypes.SimpleNamespace(
            models=pytypes.SimpleNamespace(generate_content=Mock(return_value=repeating_response))
        )
        with patch.object(agent, "client", fake_client), patch("builtins.print"):
            result = agent.run_bounded_agent("Check CASE-1045", student_id="STU-2026-001",
                                             providers=[agent.GeminiProvider()])

        tool_events = [event for event in result["trace"] if event["event"] == "tool_call"]
        self.assertEqual(len(tool_events), agent.MAX_ITERATIONS)
        self.assertEqual(fake_client.models.generate_content.call_count, agent.MAX_ITERATIONS)
        self.assertEqual(result["trace"][-1]["reason"], "Maximum iteration limit reached.")
        self.assertIn("maximum number", result["response"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
