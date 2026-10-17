"""
main.py

Runs the Week 5 execution traces through the agent loop.

The loop itself lives in agent_loop.py (one place, one MAX_TOOL_ROUNDS).
This file only sends messages in and prints what happened.

    python main.py             # the three traces from the Agent Task Contract (s9)
    python main.py --extras    # also the optional extras (escalated case, grade change,
                               # status conflict)
    python main.py --memory    # the Week 6 persistent-memory demonstration

Needs GROQ_API_KEY (primary) and GEMINI_API_KEY (fallback) in .env.
"""

import json
import sys
import uuid

from agent_loop import MAX_TOOL_ROUNDS, run_turn
from memory import default_store

# In the real app the student ID comes from the logged-in session, never from the model.
# Only the memory demo uses one; the Week 5 traces run without it, so they never touch memory.
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


def run_conversation(title: str, messages: list, student_id=None) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    conversation_id = f"demo-{uuid.uuid4().hex[:8]}"  # a new conversation every time
    for number, message in enumerate(messages, start=1):
        result = run_turn(message, conversation_id, student_id=student_id)
        print_turn(number, message, result)


def run_traces(traces: list) -> None:
    for title, messages in traces:
        run_conversation(title, messages)


def show_memory(store, label: str) -> None:
    print(f"\n>>> Everything stored about the demo student ({label}):")
    print(json.dumps(store.export_all(DEMO_STUDENT_ID), indent=2))


def run_memory_demo() -> None:
    """
    Week 6: memory improves a legitimate task (no retyping the case ID, a warning about
    a duplicate ticket) without silently controlling a decision: a remembered ID cannot
    open a ticket, and the student can override the duplicate warning.

    Needs MEMORY_PEPPER in .env (a random secret of 16+ characters).
    """
    store = default_store()
    store.revoke_consent(DEMO_STUDENT_ID)  # clean slate: withdraws consent and deletes any data
    print("Memory policy:", json.dumps(store.policy(), indent=2))

    print("\n>>> The student opts in through the app. The model can never do this.")
    store.grant_consent(DEMO_STUDENT_ID)

    run_conversation(
        "Conversation 1: first visit. The student opens a ticket. Memory is saved from the tool results.",
        ["Please open a registration ticket for CASE-1045. My registered course units are missing."],
        student_id=DEMO_STUDENT_ID,
    )
    show_memory(store, "after conversation 1")

    run_conversation(
        "Conversation 2: a NEW conversation. Memory lets the student ask about 'my case' with no ID. "
        "A ticket still needs the student to give the case ID, and a duplicate is flagged first.",
        [
            "What is the status of my case?",
            "Please open a ticket for it. The course units are still missing.",
            "Yes, it is CASE-1045.",
            "Yes, I still need a new ticket for CASE-1045.",
        ],
        student_id=DEMO_STUDENT_ID,
    )

    print("\n>>> Audit trail (who did what, when; never the content):")
    for row in store.audit_trail(DEMO_STUDENT_ID):
        print(f"  {row['at']}  {row['actor']:<8} {row['action']:<16} {row['detail']}")

    print("\n>>> The student withdraws consent. The application deletes the data:")
    print("records deleted:", store.revoke_consent(DEMO_STUDENT_ID))
    show_memory(store, "after withdrawing consent")

    run_conversation(
        "Conversation 3: after withdrawal, nothing is remembered and the agent asks for the case ID.",
        ["What is the status of my case?"],
        student_id=DEMO_STUDENT_ID,
    )


if __name__ == "__main__":
    if "--memory" in sys.argv:
        run_memory_demo()
    else:
        run_traces(CONTRACT_TRACES)
        if "--extras" in sys.argv:
            run_traces(EXTRA_TRACES)