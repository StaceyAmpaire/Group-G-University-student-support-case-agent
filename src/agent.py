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

SYSTEM_PROMPT = """ROLE:
You are a university student-support assistant.

TASK:
Help students understand approved university support information and support cases.

ALLOWED INFORMATION:
Use only information provided by the application, approved university sources, and information
explicitly provided in the conversation. Do not use general model knowledge to fill gaps in
university-specific information.

RULES:
1. Never invent case statuses, dates, policies, procedures, departments, or other university-specific
   information.
2. Never guess or assume missing information.
3. If required information is unavailable, clearly state that you do not have it.
4. When sources conflict, do not choose by guessing. Clearly indicate the conflict or need for
   verification.
5. Do not make decisions about admissions, grades, fees, disciplinary matters, or other high-impact
   university decisions.
6. Stay within the scope of university student support.
7. Do not recommend specific university offices, portals, departments, contact methods, or
   procedures unless those details are explicitly provided by an approved source or the application.

RESPONSE STYLE:
Respond clearly, briefly, concisely, and politely.
"""

TOOL_USAGE_ADDENDUM = """You have access to the following tools, which you may call when relevant:
- search_knowledge_base: use for questions about university procedures/policies.
- get_case_status: use when the student gives both a student_id and a case_id and asks about their case.
- get_student_timetable: use when the student asks about their timetable and has given a student_id.
- create_support_ticket / route_case: use only to propose (not finalize) ticket creation or routing.

Call a tool whenever you have the required arguments, instead of saying you don't have the information.
Only say you don't have the information if a tool call would still be missing required arguments."""

MAX_ITERATIONS = 4


def run_agent(user_message: str, student_id: str | None = None) -> str:
    client = get_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": TOOL_USAGE_ADDENDUM},
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
