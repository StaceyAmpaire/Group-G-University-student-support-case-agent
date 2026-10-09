from pathlib import Path
import os
import json

from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools import get_case_status, create_support_ticket


load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL = "gemini-3.7-flash"
MAX_ITERATIONS = 3

ALLOWED_TOOLS = {
    "get_case_status",
    "create_support_ticket",
}

ALLOWED_CATEGORIES = {
    "registration",
    "fees",
    "it_support",
    "accommodation",
    "academic_records",
    "other",
}

SYSTEM_INSTRUCTION = """
You are a bounded University Student-Support Case Agent.

GOAL:
Help students follow up on existing support cases.

AVAILABLE TOOLS:
1. get_case_status: retrieve a case's current information.
2. create_support_ticket: create a simulated support ticket.

WORKFLOW:
- Check the case before deciding whether a ticket is justified.
- Use one tool at a time.
- Consider the actual tool result before deciding the next action.
- - Create a follow-up ticket only when the student explicitly
  requests one, the case was successfully checked, and the
  student provides a clear issue requiring follow-up.
- Do not create a duplicate ticket automatically just because
  a case is already in progress.
- Do not invent case IDs, tool results, or successful actions.
- If a case cannot be found, explain the problem and ask the
  student to verify the case ID.
- If information is missing, ask the student for it.
- Never change grades, fees, admissions, or official records.
- Never claim an action succeeded unless the tool confirms it.
- If a tool fails, stop or recover safely instead of guessing.
- If no available tool is appropriate, explain the limitation.
- Respond briefly and clearly.
- Do not recommend university offices, contacts, portals, or
  procedures unless they are explicitly provided by an approved
  source or the application.
"""


TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="get_case_status",
        description="Retrieve the current status and details of an existing support case.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "case_id": types.Schema(
                    type="STRING",
                    description="Existing support case ID, for example CASE-1045."
                )
            },
            required=["case_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="create_support_ticket",
        description="Create a simulated support ticket for an existing case that needs follow-up.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "case_id": types.Schema(
                    type="STRING",
                    description="Existing support case ID."
                ),
                "category": types.Schema(
                    type="STRING",
                    description="Issue category: registration, fees, it_support, accommodation, academic_records, or other."
                ),
                "description": types.Schema(
                    type="STRING",
                    description="Description of the student's issue."
                ),
            },
            required=["case_id", "category", "description"],
        ),
    ),
]

TOOLS = [
    types.Tool(function_declarations=TOOL_DECLARATIONS)
]


def execute_tool(name, args, state, student_message):
    """Authorize, validate and execute one approved tool."""

    if name not in ALLOWED_TOOLS:
        return {
            "success": False,
            "error": "Tool is not authorized."
        }

    if name == "get_case_status":
        case_id = str(args.get("case_id", "")).strip()

        if not case_id:
            return {
                "success": False,
                "error": "case_id is required."
            }

        result = get_case_status(case_id)

        if result.get("success"):
            state["checked_case_id"] = case_id
            state["case_result"] = result

        return result

    if name == "create_support_ticket":
        case_id = str(args.get("case_id", "")).strip()
        category = str(args.get("category", "")).strip()
        description = str(args.get("description", "")).strip()

        # Require explicit student permission to create a ticket.
        message = student_message.lower()
        explicit_request = any(
            phrase in message
            for phrase in (
                "create a support ticket",
                "create a ticket",
                "open a ticket",
                "raise a ticket",
                "submit a ticket",
            )
        )

        if not explicit_request:
            return {
                "success": False,
                "error": "The student has not explicitly requested ticket creation."
            }

        # A case must have been checked successfully first.
        if not state.get("case_result", {}).get("success"):
            return {
                "success": False,
                "error": "Check the existing case successfully before creating a ticket."
            }

        if case_id != state.get("checked_case_id"):
            return {
                "success": False,
                "error": "The ticket case ID must match the checked case."
            }

        if state["case_result"].get("status") not in {
    "Under Review",
    "in_progress",
    "In Progress",
}:
            return {
                "success": False,
                "error": "A ticket cannot be created by this workflow unless the case is Under Review."
            }

        if not category or category.lower() not in ALLOWED_CATEGORIES:
            return {
                "success": False,
                "error": "A valid support category is required."
            }

        if not description:
            return {
                "success": False,
                "error": "A description is required."
            }

        # Prevent repeated ticket creation in the same run.
        if state.get("ticket_created"):
            return {
                "success": False,
                "error": "A ticket has already been created in this run."
            }

        result = create_support_ticket(
            case_id,
            category,
            description
        )

        if result.get("success"):
            state["ticket_created"] = True

        return result


def run_bounded_agent(student_message):
    """Run a goal-directed workflow with a strict tool-call limit."""

    state = {
        "checked_case_id": None,
        "case_result": None,
        "ticket_created": False,
    }

    trace = []
    history = [student_message]

    print("\nStudent:", student_message)

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n[ITERATION {iteration}/{MAX_ITERATIONS}]")

        response = client.models.generate_content(
            model=MODEL,
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=TOOLS,
            ),
        )

        function_calls = response.function_calls

        if not function_calls:
            final_text = response.text or (
                "I could not produce a final response."
            )

            trace.append({
                "iteration": iteration,
                "event": "stop",
                "reason": "The model did not request another tool.",
                "response": final_text,
            })

            print("[STOP] No further tool call requested.")
            print("Agent:", final_text)

            return {
                "response": final_text,
                "trace": trace,
                "state": state,
            }

        # The system must act on one tool at a time.
        if len(function_calls) > 1:
            final_text = (
                "I stopped because the model requested multiple tools "
                "at once. Please retry the request."
            )

            trace.append({
                "iteration": iteration,
                "event": "stop",
                "reason": "Multiple simultaneous tool calls rejected.",
            })

            print("[STOP] Multiple tool calls rejected.")
            print("Agent:", final_text)

            return {
                "response": final_text,
                "trace": trace,
                "state": state,
            }

        function_call = function_calls[0]
        name = function_call.name
        args = dict(function_call.args or {})

        print("[TOOL REQUEST]", name)
        print("[ARGUMENTS]", json.dumps(args, indent=2))

        result = execute_tool(
            name,
            args,
            state,
            student_message,
        )

        print("[TOOL RESULT]", json.dumps(result, indent=2))

        trace.append({
            "iteration": iteration,
            "event": "tool_call",
            "tool": name,
            "arguments": args,
            "result": result,
        })

        # Keep Gemini's tool request in the conversation history.
        history.append(response.candidates[0].content)

        # Return the real application result to Gemini.
        history.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=name,
                        response=result,
                    )
                ],
            )
        )

    # Hard stop: no fourth tool call is permitted.
    final_message = (
        "I stopped because the maximum number of tool iterations "
        "was reached. Please review the available results or "
        "continue with a new request."
    )

    trace.append({
        "event": "stop",
        "reason": "Maximum iteration limit reached.",
    })

    print("\n[STOP] Maximum iteration limit reached.")
    print("Agent:", final_message)

    return {
        "response": final_message,
        "trace": trace,
        "state": state,
    }


def run_demo(title, message):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    result = run_bounded_agent(message)

    print("\nEXECUTION TRACE")
    print(json.dumps(result["trace"], indent=2))

    return result



if __name__ == "__main__":

    results = []

    results.append(run_demo(
        "TRACE 1: CASE CHECK AND TICKET CREATION",
       "Check CASE-1045. Course registration is not reflecting "
"on my portal. Please create a follow-up support ticket "
"for this registration issue."
    ))

    results.append(run_demo(
        "TRACE 2: UNKNOWN CASE AND SAFE RECOVERY",
        "Check CASE-9999. If it exists and is Under Review, "
        "create a support ticket. If the case cannot be found, "
        "tell me to verify its ID."
    ))

    results.append(run_demo(
        "TRACE 3: OUT-OF-SCOPE REQUEST",
        "Change my official exam grade to 80%."
    ))

    evidence_dir = Path("docs/evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)

    trace_file = evidence_dir / "week5_execution_traces.json"
    trace_file.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\nExecution traces saved to: {trace_file}")

