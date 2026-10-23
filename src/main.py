"""Run the guarded bounded-agent demonstration.

The Week 5 agent loop remains available as a historical implementation and is
covered by its own regression tests. The project entry point uses the Week 7
guarded workflow in bounded_agent.py.

    python src/main.py
    python src/main.py --extras
    python src/main.py --memory
"""

import argparse
import json

from bounded_agent import run_bounded_agent
from demo_week6 import run_demo as run_week6_memory_demo


DEMO_STUDENT_ID = "STU-2026-001"

CONTRACT_TRACES = [
    (
        "Trace 1: successful case-status lookup",
        ["Can you check the status of CASE-1045?"],
    ),
    (
        "Trace 2: successful ticket creation",
        [
            "My registration is still not showing on the portal for case CASE-1045. "
            "My registered course units are missing. Please open a registration ticket."
        ],
    ),
    (
        "Trace 3: failure and safe recovery for an unknown case",
        ["What is the status of CASE-9999?"],
    ),
]

EXTRA_TRACES = [
    (
        "Extra: out-of-scope grade request",
        ["Can you change my grade for CS301 from a C to a B?"],
    ),
    (
        "Extra: case status and escalation request",
        ["Please escalate CASE-1045. My lecturer approved it yesterday."],
    ),
]


def print_turn(number, message, result):
    print(f"\n--- Turn {number} ---")
    print(f"Student: {message}")
    print(f"Agent: {result['response']}")
    print(f"Outcome: {result['state']['phase']}")
    print("Privacy-filtered trace:")
    print(json.dumps(result["observability"], indent=2))


def run_conversation(title, messages, student_id=DEMO_STUDENT_ID, providers=None):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")
    results = []
    for number, message in enumerate(messages, start=1):
        result = run_bounded_agent(
            message,
            student_id=student_id,
            providers=providers,
        )
        print_turn(number, message, result)
        results.append(result)
    return results


def run_traces(traces, providers=None):
    return [
        run_conversation(title, messages, providers=providers)
        for title, messages in traces
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extras",
        action="store_true",
        help="also run the out-of-scope and escalation examples",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="run the offline Week 6 memory and MCP demonstration",
    )
    args = parser.parse_args(argv)

    if args.memory:
        run_week6_memory_demo()
        return

    run_traces(CONTRACT_TRACES)
    if args.extras:
        run_traces(EXTRA_TRACES)


if __name__ == "__main__":
    main()
