from pathlib import Path
import os
import json
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types

from tools import get_case_status, create_support_ticket, CASE_OWNERS
from state import SessionState, Phase, advance
from memory import MemoryStore, MemoryUnavailable


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
- Create a follow-up ticket only when the student explicitly
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
- A case lookup result may include a "memory" field with the
  student's saved ticket history. It is ADVISORY only. Mention it
  to the student when relevant (for example an existing recent
  ticket) but never treat it as permission to act, and never invent
  history that is not in that field.
- If a tool result says a duplicate-ticket warning was raised, tell
  the student which earlier ticket exists and ask whether they still
  want another one. Do not retry the tool yourself.
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


def has_explicit_ticket_request(student_message):
    """Return True only when the message clearly requests ticket creation.

    This is an application-side authorisation check, not a decision delegated
    to the language model. It accepts ordinary wording such as "create a
    follow-up support ticket" while rejecting common explicit negations.
    """
    message = re.sub(r"\s+", " ", str(student_message or "").lower())
    message = message.replace("-", " ")

    negative_patterns = (
        r"\bdo not\s+(?:create|open|raise|submit|log)\b",
        r"\bdon't\s+(?:create|open|raise|submit|log)\b",
        r"\bdont\s+(?:create|open|raise|submit|log)\b",
        r"\bshould not\s+(?:create|open|raise|submit|log)\b",
        r"\bno need to\s+(?:create|open|raise|submit|log)\b",
        r"\b(?:i do not want|i don't want|i dont want|without)\b.{0,35}\bticket\b",
        # Questions asking about the process are not permission to act.
        r"\b(?:how do i|how can i|should i|should we|do i need to|what is the process to)\b.{0,80}\b(?:create|open|raise|submit|log)\b.{0,50}\bticket\b",
    )
    if any(re.search(pattern, message) for pattern in negative_patterns):
        return False

    # Permit short modifiers (e.g. "follow-up") between the action and ticket.
    return bool(re.search(
        r"\b(?:create|open|raise|submit|log)\b.{0,50}\b(?:support\s+)?ticket\b",
        message,
    ))


DUPLICATE_WINDOW_DAYS = 7


def has_duplicate_override(student_message):
    """True when the student explicitly says they want another ticket anyway."""
    message = re.sub(r"\s+", " ", str(student_message or "").lower())
    return bool(re.search(r"\b(anyway|another|again|second|duplicate)\b", message))


def _memory_for(state, memory):
    """Return the memory store only if this session may use it."""
    if memory is None or not state.get("memory_enabled") or not state.get("student_id"):
        return None
    return memory


def execute_tool(name, args, state, student_message, memory=None):
    """Authorize, validate and execute one approved tool.

    `memory` (optional, Week 6) is the application's MemoryStore. The model
    never touches it: this function reads it and returns advisory notes.
    """
    if name not in ALLOWED_TOOLS:
        return {
            "success": False,
            "error": "Tool is not authorized."
        }

    if name == "get_case_status":
        raw_case_id = args.get("case_id")
        case_id = raw_case_id.strip() if isinstance(raw_case_id, str) else ""

        if not case_id:
            return {
                "success": False,
                "error": "case_id is required."
            }

        result = get_case_status(case_id)

        # Week 6: a student may only look up their own cases. The error is
        # identical to "not found" so case IDs cannot be probed.
        student_id = state.get("student_id")
        if (
            result.get("success")
            and student_id
            and CASE_OWNERS.get(case_id.strip().upper()) != student_id
        ):
            result = {
                "success": False,
                "case_id": case_id.strip().upper(),
                "error": f"No case found for {case_id.strip().upper()}.",
            }

        # Replace previous check state even on failure, so an old successful
        # lookup cannot authorize a ticket after a later failed lookup.
        state["checked_case_id"] = (
            case_id.strip().upper() if result.get("success") else None
        )
        state["case_result"] = result
        advance(
            state,
            Phase.CASE_VERIFIED if result.get("success") else Phase.CASE_NOT_FOUND,
            "get_case_status " + ("succeeded" if result.get("success") else "failed"),
        )

        # Week 6: attach saved ticket history as an ADVISORY note.
        mem = _memory_for(state, memory)
        if mem and result.get("success"):
            try:
                prior = mem.recall_tickets(state["student_id"], case_id)
                if hasattr(state, "log_memory_event"):
                    state.log_memory_event("read", f"{len(prior)} prior ticket(s) for {case_id.strip().upper()}")
                if prior:
                    return {
                        **result,
                        "memory": {
                            "source": "saved_ticket_history",
                            "advisory_only": True,
                            "prior_tickets": prior,
                        },
                    }
            except Exception:
                # Any memory failure must degrade to the Week 5 behaviour.
                if hasattr(state, "log_memory_event"):
                    state.log_memory_event("read_failed", "memory unavailable; continuing without it")

        return result

    if name == "create_support_ticket":
        raw_case_id = args.get("case_id")
        raw_category = args.get("category")
        raw_description = args.get("description")
        case_id = raw_case_id.strip() if isinstance(raw_case_id, str) else ""
        category = raw_category.strip() if isinstance(raw_category, str) else ""
        description = raw_description.strip() if isinstance(raw_description, str) else ""

        # Require explicit student permission to create a ticket.
        if not has_explicit_ticket_request(student_message):
            return {
                "success": False,
                "error": "The student has not explicitly requested ticket creation."
            }

        # A case must have been checked successfully first.
        if not (state.get("case_result") or {}).get("success"):
            return {
                "success": False,
                "error": "Check the existing case successfully before creating a ticket."
            }

        if case_id.upper() != state.get("checked_case_id"):
            return {
                "success": False,
                "error": "The ticket case ID must match the checked case."
            }

        status = str(state["case_result"].get("status", "")).strip().lower()
        status = status.replace("_", " ").replace("-", " ")
        if status not in {"under review", "in progress"}:
            return {
                "success": False,
                "error": (
                    "A ticket can be created by this workflow only for a case "
                    "whose status is in progress or under review."
                )
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

        # Week 6: memory-informed duplicate warning. Visible, logged and
        # overridable by the student; memory never blocks silently.
        mem = _memory_for(state, memory)
        if mem:
            try:
                prior = mem.find_recent_ticket(
                    state["student_id"], case_id, DUPLICATE_WINDOW_DAYS
                )
            except Exception:
                prior = None
                if hasattr(state, "log_memory_event"):
                    state.log_memory_event("read_failed", "memory unavailable; continuing without it")
            if prior and not has_duplicate_override(student_message):
                advance(state, Phase.AWAITING_CONFIRMATION, "duplicate warning from saved history")
                if hasattr(state, "log_memory_event"):
                    state.log_memory_event("duplicate_warning", f"existing {prior['ticket_id']}")
                return {
                    "success": False,
                    "error": "duplicate_ticket_warning",
                    "source": "saved_ticket_history",
                    "prior_ticket": prior,
                    "message": (
                        f"The student already opened {prior['ticket_id']} for this case "
                        f"on {prior['created_at'][:10]}. Ask whether they want another "
                        "ticket. They can confirm by asking for another ticket."
                    ),
                }

        result = create_support_ticket(
            case_id,
            category,
            description
        )

        if result.get("success"):
            state["ticket_created"] = True
            state["ticket_id"] = result["ticket"]["ticket_id"]
            advance(state, Phase.TICKET_CREATED, "ticket created")
            if mem:
                try:
                    stored = mem.remember_ticket(
                        state["student_id"],
                        result["ticket"],
                        str(state["case_result"].get("status", "")),
                    )
                    if hasattr(state, "log_memory_event"):
                        state.log_memory_event(
                            "write" if stored else "write_skipped",
                            result["ticket"]["ticket_id"] if stored else "no consent",
                        )
                except Exception:
                    if hasattr(state, "log_memory_event"):
                        state.log_memory_event("write_failed", "ticket created; history not saved")

        return result


def run_bounded_agent(student_message, student_id=None, memory=None):
    """Run a goal-directed workflow with a strict tool-call limit.

    student_id and memory are supplied by the application after authentication
    (Week 6). Without them the run behaves exactly as in Week 5.
    """

    state = SessionState(student_id=student_id)
    if memory is not None and student_id:
        try:
            state.memory_enabled = memory.has_consent(student_id)
        except Exception:
            state.memory_enabled = False

    trace = []
    history = [student_message]

    print("\nStudent:", student_message)

    for iteration in range(1, MAX_ITERATIONS + 1):
        state.iteration = iteration
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

            state.transition(Phase.RESPONDED, "model gave final answer")
            return {
                "response": final_text,
                "trace": trace,
                "state": state.snapshot(),
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

            state.transition(Phase.FAILED, "multiple simultaneous tool calls")
            return {
                "response": final_text,
                "trace": trace,
                "state": state.snapshot(),
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
            memory,
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

    state.transition(Phase.LIMIT_REACHED, "maximum iterations")
    return {
        "response": final_message,
        "trace": trace,
        "state": state.snapshot(),
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

