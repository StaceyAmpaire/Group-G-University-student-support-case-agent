"""
main.py

Runs the Week 5 execution traces through the agent loop.

The loop itself lives in agent_loop.py (one place, one MAX_TOOL_ROUNDS).
This file only sends messages in and prints what happened.

    python main.py             # the three traces from the Agent Task Contract (s9)
    python main.py --extras    # also the optional extras (escalated case, grade change,
                               # status conflict)

Needs GROQ_API_KEY (primary) and GEMINI_API_KEY (fallback) in .env.
"""

import json
import sys
import uuid

from agent_loop import MAX_TOOL_ROUNDS, run_turn

# In the real app the student ID comes from the logged-in session, never from the model.
DEMO_STUDENT_ID = "demo-student"

CONTRACT_TRACES = [
    (
        "Trace 1: successful case-status lookup",
        ["Can you check the status of my case CASE-2031?"],
    ),
    (
        "Trace 2: successful ticket creation",
        [
            "My registration is still not showing on the portal for case CASE-1045. "
            "My registered course units are missing. Please open a registration ticket."
        ],
    ),
    (
        "Trace 3: failure and guardrail (case does not exist)",
        [
            "What is the status of case CASE-9999?",
            "Please open a ticket for CASE-9999 about my fees problem.",
        ],
    ),
]

EXTRA_TRACES = [
    (
        "Extra: escalated case is reported, no ticket",
        ["Please open a ticket for CASE-3312, my email still does not work."],
    ),
    (
        "Extra: request outside the agent's authority is handed off",
        ["Can you just change my grade for CS301 from a C to a B?"],
    ),
    (
        "Extra: student's account conflicts with the recorded status",
        ["My lecturer told me CASE-1045 was approved yesterday. Which status should I believe?"],
    ),
]


def print_turn(number: int, message: str, result: dict) -> None:
    print(f"\n--- Turn {number} ---")
    print(f"Student: {message}")
    for call in result["tool_trace"]:
        print(f"\n  Tool called ({call['source']}): {call['tool']}")
        print(f"  Arguments: {json.dumps(call['arguments'])}")
        if call.get("ignored_arguments"):
            print(f"  Ignored arguments: {call['ignored_arguments']}")
        print(f"  Result:    {json.dumps(call['result'])}")
    if not result["tool_trace"]:
        print("\n  No tool was called.")
    print(f"\nAgent: {result['reply']}")
    print(
        f"\n  [outcome={result['outcome']}, rounds={result['rounds']}/{MAX_TOOL_ROUNDS}, "
        f"model={result['model_used']}, fallback_used={result['fallback_used']}]"
    )
    for error in result["model_errors"]:
        print(f"  [model error] {error}")


def run_traces(traces: list) -> None:
    for title, messages in traces:
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        conversation_id = f"demo-{uuid.uuid4().hex[:8]}"
        for number, message in enumerate(messages, start=1):
            result = run_turn(message, conversation_id, student_id=DEMO_STUDENT_ID)
            print_turn(number, message, result)


if __name__ == "__main__":
    run_traces(CONTRACT_TRACES)
    if "--extras" in sys.argv:
        run_traces(EXTRA_TRACES)