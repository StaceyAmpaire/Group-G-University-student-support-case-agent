import os
import json

from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools import TOOL_SCHEMAS, TOOL_FUNCTIONS


load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


system_instruction = """
You are a University Student Support Assistant.

TASK:
Help students understand approved university support information
and information about their support cases.

ALLOWED INFORMATION:
Use only information provided by the application,
approved university sources, and information explicitly
provided in the conversation.

Do not use general model knowledge to fill gaps in
university-specific information.

RULES:
1. Never invent case statuses, dates, policies, procedures,
   departments, or other university-specific information.

2. Never guess or assume missing information. This includes
   information required to call a tool: if a required field
   (e.g. a case ID, student name, issue category, or
   description) is missing, ask the student for it instead of
   guessing or calling the tool with incomplete information.

3. If the information required to answer the question is
   unavailable, clearly state that you do not have that information.

4. When sources provide conflicting information, do not choose
   an answer by guessing. Clearly indicate that the information
   conflicts or requires verification.

5. Do not make decisions about admissions, grades, fees,
   disciplinary matters, or other high-impact university decisions.
   You may not escalate, close, or resolve a case yourself --
   only a human staff member can do that.

6. Stay within the scope of university student support.

7. Do not recommend specific university offices, portals,
   departments, contact methods, or procedures unless those
   details are explicitly provided by an approved source
   or the application.

TOOLS:
You have two tools available:
  - check_case_status: use this when a student asks about the
    status or details of an existing case by its case ID.
  - create_support_ticket: use this when a student reports a
    new issue that is not already resolved by the information
    available to you. Only call it once you have the student's
    name, a category, and a description -- per Rule 2, ask for
    anything missing rather than calling the tool without it.

RESPONSE STYLE:
Respond clearly, briefly, concisely, and politely.
"""


GEMINI_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name=schema["name"],
            description=schema["description"],
            parameters_json_schema=schema["parameters"],
        )
        for schema in TOOL_SCHEMAS
    ]
)

MAX_TOOL_ROUNDS = 3


def run_agent_turn(student_message: str) -> dict:
    """
    Runs one full agent turn, including any tool calls Gemini requests.
    Returns a trace dict (tool calls, arguments, results, final response)
    so each step can be inspected/screenshotted as the Week 4 task asks.
    """
    trace = {"student_message": student_message, "tool_calls": []}

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=student_message)],
        )
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.models.generate_content(
            model="gemini-3.7-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[GEMINI_TOOL],
            ),
        )

        candidate_parts = response.candidates[0].content.parts
        function_call_parts = [
            part for part in candidate_parts if part.function_call is not None
        ]

        if not function_call_parts:
            trace["final_response"] = response.text
            return trace

        # Echo the model's own turn (including its function_call parts)
        # back into the conversation before adding our function results.
        contents.append(response.candidates[0].content)

        response_parts = []
        for part in function_call_parts:
            function_call = part.function_call
            tool_name = function_call.name
            tool_args = dict(function_call.args) if function_call.args else {}

            tool_function = TOOL_FUNCTIONS.get(tool_name)
            if tool_function is None:
                tool_result = {"error": f"Unknown tool '{tool_name}' requested."}
            else:
                tool_result = tool_function(**tool_args)

            trace["tool_calls"].append(
                {"tool": tool_name, "args": tool_args, "result": tool_result}
            )

            response_parts.append(
                types.Part.from_function_response(name=tool_name, response=tool_result)
            )

        contents.append(types.Content(role="user", parts=response_parts))

    trace["final_response"] = (
        "I wasn't able to complete this after several tool calls. "
        "Please rephrase your request or contact support directly."
    )
    return trace


# ---------------------------------------------------------------------------
# Test cases: the original conflicting-information example, plus the tool
# cases the Week 4 task asks for (known case, unknown case, complete ticket,
# incomplete ticket, out-of-scope request).
# ---------------------------------------------------------------------------

TEST_MESSAGES = [
    # Original conflict-handling example (no tool call expected -- this
    # tests Rule 4 directly, using case info supplied inline rather than
    # via the tool).
    """
APPLICATION CASE INFORMATION:
Case ID: CASE-1045
Status: Under Review
Last Updated: September 7, 2026

STUDENT MESSAGE:
My lecturer told me that CASE-1045 was approved yesterday.

Which status should I believe?
""",

    # Known case ID -> check_case_status should find it
    "Can you check the status of my case CASE-1045?",

    # Unknown case ID -> check_case_status should return not-found gracefully
    "What's the status of case CASE-9999?",

    # Complete information -> create_support_ticket should succeed
    (
        "My name is Brian Okello. I can't access the student portal to "
        "register for my courses this semester. Please open a ticket for "
        "me under registration."
    ),

    # Missing information -> agent should ask for the missing field
    # rather than call the tool with incomplete data
    "I have a problem with my fees. Can you open a support ticket?",

    # Out-of-scope request -> agent should decline per Rule 5
    "Can you just change my grade for CS301 from a C to a B?",
]


if __name__ == "__main__":
    for i, message in enumerate(TEST_MESSAGES, start=1):
        print("=" * 80)
        print(f"TEST {i}")
        print(f"Student: {message.strip()}")
        print("-" * 80)

        result = run_agent_turn(message)

        if result["tool_calls"]:
            for call in result["tool_calls"]:
                print(f"\nTool called: {call['tool']}")
                print(f"Arguments:   {json.dumps(call['args'], indent=2)}")
                print(f"Result:      {json.dumps(call['result'], indent=2)}")
        else:
            print("\nNo tool was called for this turn.")

        print(f"\nFinal response:\n{result['final_response']}\n")