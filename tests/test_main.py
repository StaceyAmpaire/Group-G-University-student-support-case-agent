"""Offline tests for the guarded project entry point."""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import test_bounded_agent  # noqa: F401  (provides the google-genai test stub if needed)
import main
import tools
from bounded_agent import ModelResponse, ToolRequest


class ScriptedProvider:
    name = "groq"
    model = "scripted"

    def __init__(self, *responses):
        self.responses = list(responses)

    def generate(self, history):
        response = self.responses.pop(0)
        if isinstance(response, ToolRequest):
            return ModelResponse("", [response])
        return ModelResponse(response, [])


class ExplodingProvider:
    name = "groq"
    model = "must-not-run"

    def generate(self, history):
        raise AssertionError("the provider should not be called")


class MainEntryPointTests(unittest.TestCase):
    def setUp(self):
        tools.TICKET_STORE.clear()

    def test_main_demo_uses_identity_guard_before_model_call(self):
        provider = ExplodingProvider()
        with patch("builtins.print"):
            (result,) = main.run_conversation(
                "Unauthenticated test",
                ["Check CASE-1045."],
                student_id=None,
                providers=[provider],
            )

        self.assertEqual(result["state"]["phase"], "responded")
        self.assertIn("until you are signed in", result["response"])
        self.assertFalse(any(event["event"] == "model_call" for event in result["trace"]))

    def test_main_demo_runs_the_bounded_case_lookup_workflow(self):
        provider = ScriptedProvider(
            ToolRequest("get_case_status", {"case_id": "CASE-1045"}, "call-1"),
            "CASE-1045 is in progress.",
        )
        with patch("builtins.print"):
            (result,) = main.run_conversation(
                "Case-status test",
                ["Check CASE-1045."],
                providers=[provider],
            )

        self.assertEqual(result["state"]["phase"], "responded")
        self.assertIn("in progress", result["response"])
        self.assertEqual(result["trace"][0]["event"], "model_call")
        self.assertEqual(result["trace"][1]["event"], "tool_call")

    def test_demo_scripts_are_safe_to_import(self):
        script = (
            "import sys; sys.path.insert(0, 'src'); "
            "import demo_week6, evaluate_prompt, rag_demo"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "")


if __name__ == "__main__":
    unittest.main()
