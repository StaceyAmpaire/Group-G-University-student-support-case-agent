"""
Bounded agent loop for the University Student-Support Case Agent.

Pattern (from the charter): Sense/Context -> Plan/Decide -> Act/Tool
-> Observe -> Stop or Re-plan.

This is intentionally small: one model call to decide on a tool (or
answer directly), one stub tool execution, one follow-up model call
to turn the tool result into a grounded reply. Iteration is capped
so the agent can't loop forever if something goes wrong.
"""

import json
from model_client import get_client, call_model
from tools import TOOL_SCHEMAS, TOOL_EXECUTORS

SYSTEM_PROMPT = """You are a university student-support assistant.

You may only:
- answer questions using the search_knowledge_base tool (never invent answers)
- look up case status or timetable via the provided tools, if a student_id is given
- propose creating or routing a support ticket via the provided tools

You must NEVER make admissions, grading, disciplinary, or fee decisions.
If a request involves a fee waiver, appeal, disciplinary matter, or
modifying an official record, say clearly that this requires human
staff review and do not attempt to resolve it yourself.
If you don't have enough information to call a tool, ask the student
for what's missing instead of guessing.
"""

MAX_ITERATIONS = 4


def run_agent(user_message: str, student_id: str | None = None) -> str:
    client = get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for _ in range(MAX_ITERATIONS):
        reply = call_model(client, messages, tools=TOOL_SCHEMAS)

        if not reply.tool_calls:
            # Model chose to answer directly, no tool needed. Stop.
            return reply.content

        # Act: execute each proposed tool call against the stub executors.
        serialized_tool_calls = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in reply.tool_calls
        ]
        messages.append({"role": "assistant", "content": reply.content or "", "tool_calls": serialized_tool_calls})
        for call in reply.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            executor = TOOL_EXECUTORS.get(name)
            if executor is None:
                result = {"error": f"unknown tool {name}"}
            else:
                result = executor(**args)
            # Observe: feed the tool result back to the model.
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })
        # Loop again: model re-plans given the observation.

    return "I wasn't able to resolve this within my step limit. Please try rephrasing or contact support staff directly."


if __name__ == "__main__":
    # Sample scenario from the Project Charter, section 4.
    sample = "I reported a problem with my course registration. What does the university procedure say, and can you check whether my case has been updated?"
    print(run_agent(sample, student_id="STU001"))

    # Forces a tool call: full IDs given, so the model should call get_case_status.
    sample2 = "My student ID is STU001 and my case ID is CASE001. What's the status?"
    print(run_agent(sample2, student_id="STU001"))
