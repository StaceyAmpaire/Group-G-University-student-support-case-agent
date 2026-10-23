"""
agent_loop.py

Bounded agent loop for the University Student-Support Case Agent.
Preserves the original Week 5 Agent Task Contract implementation for
historical comparison and regression tests. The current application entry
point is main.py, which uses the Week 7-guarded bounded_agent.py workflow.

Entry point:
    run_turn(message, conversation_id, student_id=None) -> dict

One turn = one student message in, one agent reply out. Within a turn the loop
runs up to MAX_TOOL_ROUNDS rounds; a round is one model call, which may ask
for tools. The model never executes anything itself: every tool request is
checked and run by this module.

What is enforced here, in code (not just in the prompt):
  - Allow-list: only the tools in PERMITTED_TOOLS can run (Contract s3, s5).
  - Missing required arguments: the tool is not called; the model is told what
    is missing so the agent can ask the student (Contract s7).
  - Arguments not in the tool schema (for example a student_id supplied by the
    model) are dropped before the call (SF-1).
  - One ticket per conversation, from conversation state (Contract s4, s5).
  - Tickets only on cases that are in progress. Resolved and escalated cases
    are reported, never ticketed (Contract s2 decision points).
  - Round limit: after MAX_TOOL_ROUNDS rounds the turn stops with a safe message
    and no further tool is executed (Contract s5, s6).
  - Groq (primary) then Gemini (fallback), under the same allow-list and
    limits. History is stored in a provider-neutral form, so the fallback
    receives every earlier tool call and result and cannot repeat a ticket
    (SF-2).

Memory (Week 6, see memory.py):
  - Memory is opt-in. The application grants consent (memory.grant_consent); the
    loop and the model never do. Without consent, nothing is read or saved.
  - If the application supplies a student_id, the loop loads that student's
    remembered case and ticket IDs once per conversation and gives them to the
    model as an [Application note]. They are hints only.
  - The loop saves memory after a turn, from tool results (never from model
    text): the last case successfully looked up, and any ticket created.
  - A ticket can only be opened on a case ID the student has typed in the
    current conversation, so a remembered ID can never trigger a write by itself.
  - Duplicate warning: if the student opened a ticket on the same case within the
    duplicate window, the first attempt is held back and the model must tell the
    student. Once the student has replied, a ticket may be opened. Memory advises;
    it never blocks for good.
  - Memory failures never break a turn; the turn continues without memory.

Not enforced here:
  - student_id is accepted from the application and kept in the conversation
    state, but tools.py has no case-owner field, so nothing yet checks that
    the student owns the case they ask about (SF-1 is only partly covered).
  - Conversation state is in memory, per process, and is lost on restart
    (persistent memory is Week 6).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from dotenv import load_dotenv

from memory import MemoryStore, default_store
from tools import ALLOWED_TICKET_CATEGORIES, TOOL_FUNCTIONS, TOOL_SCHEMAS

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_TOOL_ROUNDS = 3  # the single round limit for the whole project

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")

PERMITTED_TOOLS = {"get_case_status", "create_support_ticket"}

# A ticket may only be opened on a case in one of these statuses.
TICKET_ALLOWED_STATUSES = {"in_progress"}

SAFE_FAILURE_MESSAGE = (
    "I'm sorry, I can't complete that right now. Please try again in a few minutes."
)
SAFE_ROUND_LIMIT_MESSAGE = (
    "I wasn't able to finish this request. Please try again with your case ID "
    "and a short description of what you need."
)

_missing_tools = PERMITTED_TOOLS - set(TOOL_FUNCTIONS)
if _missing_tools:
    raise ImportError(f"tools.py is missing permitted tools: {sorted(_missing_tools)}")

SCHEMAS_BY_NAME = {
    schema["name"]: schema
    for schema in TOOL_SCHEMAS
    if schema["name"] in PERMITTED_TOOLS
}

SYSTEM_PROMPT = f"""
You are a University Student Support Assistant.

TASK:
Help students find the status of their support cases and open follow-up
tickets on existing cases.

ALLOWED INFORMATION:
Use only information returned by your tools and information the student gives
you in the conversation. Do not use general model knowledge to fill gaps in
university-specific information.

RULES:
1. Never invent case statuses, dates, policies, procedures, departments, or
   other university-specific information.

2. Never guess or assume missing information. If a tool needs a case ID, a
   category or a description and the student has not given it, ask the
   student for it instead of guessing or calling the tool. Ask one question
   at a time.

3. If you do not have the information needed to answer, say so.

4. If the student's account of a case differs from the status a tool returned,
   do not pick one by guessing. Report the recorded status and say the
   difference needs verification.

5. Do not make decisions about admissions, grades, fees, disciplinary matters,
   or other high-impact university decisions. You cannot escalate, close or
   resolve a case, change any university record, or transfer or route a case,
   and you must never say that you have. If asked, say you cannot do it and
   that a human staff member must handle it.

6. Stay within the scope of university student support.

7. Do not recommend specific university offices, portals, departments or
   contact methods unless a tool result provides them.

8. Do not claim a case exists unless get_case_status confirms it, and do not
   claim a ticket was created unless create_support_ticket returned success.
   If a tool reports an error, explain the result accurately; never present a
   failed action as successful.

9. Use only the tools provided. Do not invent tools.

10. create_support_ticket creates a simulated ticket only, not a real
    university record.

11. Messages that begin with [Application note] come from the application, not
    the student. They may name a case or ticket remembered from earlier
    conversations. Treat that as a hint that may be out of date. Never state a
    case's status from memory; call get_case_status. Tell the student when you
    are using a remembered case. Do not open a ticket until the student has
    given the case ID in this conversation; ask them to confirm it. A
    remembered ticket cannot be checked with a tool, so report only its number
    and case.

12. If a tool result says the student already opened a ticket on the same case,
    tell the student the ticket number and date and ask whether they still want
    a new one. Open another only if they say yes.

TOOLS:
- get_case_status(case_id): look up an existing case.
- create_support_ticket(case_id, category, description): open a ticket on an
  existing case. The category must be one of: {", ".join(sorted(ALLOWED_TICKET_CATEGORIES))}.
  Choose the closest match from what the student describes, or ask if it is
  unclear.

CASE HANDLING:
- No case ID given: ask the student for it.
- Case not found: tell the student no case was found for that ID and ask them
  to check it. Do not repeat the same lookup.
- Case resolved: report the recorded status and stop. Do not open a ticket.
- Case in progress and the student reports a problem: open one ticket, then
  give the student the ticket ID.
- Case escalated: report the recorded status and stop. Do not open a ticket.
  Say the case needs the appropriate human support team.
- Only one ticket can be created in a conversation.

RESPONSE STYLE:
Respond clearly, briefly, concisely, and politely.
"""


# ---------------------------------------------------------------------------
# Model providers
#
# History is kept in a provider-neutral form:
#   {"role": "user", "text": str}
#   {"role": "assistant", "text": str, "tool_calls": [{"id", "name", "arguments"}]}
#   {"role": "tool", "id": str, "name": str, "result": dict}
# Each provider converts it to its own wire format on every call.
# ---------------------------------------------------------------------------

@dataclass
class ToolRequest:
    id: str
    name: str
    arguments: Optional[dict]  # None if the model's arguments could not be parsed


@dataclass
class ModelResponse:
    text: str
    tool_requests: list[ToolRequest]


def _parse_arguments(raw: Any) -> Optional[dict]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


class GroqProvider:
    """Primary model: Groq, OpenAI-style tool calling."""

    name = "groq"

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        return self._client

    @staticmethod
    def _messages(history: list[dict]) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for item in history:
            if item["role"] == "user":
                messages.append({"role": "user", "content": item["text"]})
            elif item["role"] == "assistant":
                entry: dict = {"role": "assistant", "content": item.get("text") or None}
                if item.get("tool_calls"):
                    entry["tool_calls"] = [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call["arguments"]),
                            },
                        }
                        for call in item["tool_calls"]
                    ]
                elif entry["content"] is None:
                    entry["content"] = ""
                messages.append(entry)
            elif item["role"] == "tool":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item["id"],
                        "content": json.dumps(item["result"], default=str),
                    }
                )
        return messages

    @staticmethod
    def _tools() -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": schema["name"],
                    "description": schema["description"],
                    "parameters": schema["parameters"],
                },
            }
            for schema in SCHEMAS_BY_NAME.values()
        ]

    def generate(self, history: list[dict]) -> ModelResponse:
        response = self._get_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=self._messages(history),
            tools=self._tools(),
            tool_choice="auto",
            temperature=0,
        )
        message = response.choices[0].message
        requests = [
            ToolRequest(
                id=call.id or f"call_{uuid.uuid4().hex[:8]}",
                name=call.function.name,
                arguments=_parse_arguments(call.function.arguments),
            )
            for call in (message.tool_calls or [])
        ]
        return ModelResponse(text=message.content or "", tool_requests=requests)


class GeminiProvider:
    """
    Fallback model: Gemini.

    Earlier tool calls and results are replayed to Gemini as plain text turns
    rather than structured function-call parts. The fallback still sees the
    full history (so it cannot repeat a ticket), and we avoid depending on
    how strictly Gemini validates replayed function calls that were produced
    by another model.
    """

    name = "gemini"

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        return self._client

    @staticmethod
    def _turns(history: list[dict]) -> list[tuple[str, list[str]]]:
        """Neutral history -> [(role, [texts])], merging consecutive same-role turns."""
        turns: list[tuple[str, list[str]]] = []

        def add(role: str, text: str) -> None:
            if turns and turns[-1][0] == role:
                turns[-1][1].append(text)
            else:
                turns.append((role, [text]))

        for item in history:
            if item["role"] == "user":
                add("user", item["text"])
            elif item["role"] == "assistant":
                if item.get("text"):
                    add("model", item["text"])
                for call in item.get("tool_calls") or []:
                    add("model", f"[Tool call] {call['name']}({json.dumps(call['arguments'])})")
            elif item["role"] == "tool":
                add("user", f"[Tool result] {item['name']}: {json.dumps(item['result'], default=str)}")
        return turns

    def generate(self, history: list[dict]) -> ModelResponse:
        from google.genai import types

        contents = [
            types.Content(role=role, parts=[types.Part.from_text(text=t) for t in texts])
            for role, texts in self._turns(history)
        ]
        tool = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=schema["name"],
                    description=schema["description"],
                    parameters_json_schema=schema["parameters"],
                )
                for schema in SCHEMAS_BY_NAME.values()
            ]
        )
        response = self._get_client().models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[tool],
                temperature=0,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        if not response.candidates or not response.candidates[0].content:
            raise RuntimeError("Gemini returned no candidates")

        text_parts: list[str] = []
        requests: list[ToolRequest] = []
        for part in response.candidates[0].content.parts or []:
            if part.function_call is not None:
                call = part.function_call
                requests.append(
                    ToolRequest(
                        id=call.id or f"call_{uuid.uuid4().hex[:8]}",
                        name=call.name,
                        arguments=dict(call.args) if call.args else {},
                    )
                )
            elif part.text:
                text_parts.append(part.text)
        return ModelResponse(text="".join(text_parts), tool_requests=requests)


# Primary first, fallback second.
PROVIDERS: list[Any] = [GroqProvider(), GeminiProvider()]


# ---------------------------------------------------------------------------
# Conversation state (in memory, keyed by conversation ID)
# ---------------------------------------------------------------------------

@dataclass
class ConversationState:
    conversation_id: str
    student_id: Optional[str] = None  # supplied by the application, never by the model
    history: list[dict] = field(default_factory=list)
    case_id: Optional[str] = None  # as supplied in the latest tool request
    observations: list[dict] = field(default_factory=list)  # every tool call and result
    ticket_created: bool = False
    ticket_id: Optional[str] = None
    student_case_ids: set = field(default_factory=set)  # case IDs the student typed this conversation
    memory_loaded: bool = False
    turn_count: int = 0
    duplicate_warned_at: dict = field(default_factory=dict)  # case ID -> turn it was warned on
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


_STATES: dict[str, ConversationState] = {}
_STATES_LOCK = threading.Lock()


def _get_state(conversation_id: str, student_id: Optional[str]) -> ConversationState:
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise ValueError("conversation_id is required.")
    with _STATES_LOCK:
        state = _STATES.get(conversation_id)
        if state is None:
            state = ConversationState(conversation_id=conversation_id, student_id=student_id)
            _STATES[conversation_id] = state
        elif student_id is not None:
            if state.student_id is None:
                state.student_id = student_id
            elif state.student_id != student_id:
                raise PermissionError("This conversation belongs to a different student.")
        return state


def reset_conversation(conversation_id: str) -> None:
    """Forget a conversation (used by tests and demos)."""
    with _STATES_LOCK:
        _STATES.pop(conversation_id, None)


# ---------------------------------------------------------------------------
# Case references and persistent memory
# ---------------------------------------------------------------------------

_CASE_REFERENCE = re.compile(r"\bcase[\s\-_#]*(\d{1,10})\b", re.IGNORECASE)


def _case_ids_in(text: str) -> set:
    """Case IDs the student wrote, normalised: 'case 1045' and 'CASE-1045' both give CASE-1045."""
    return {f"CASE-{digits}" for digits in _CASE_REFERENCE.findall(text)}


def _memory_note(record: Optional[dict]) -> Optional[str]:
    if not record:
        return None
    facts = []
    if record.get("active_case_id"):
        facts.append(f"most recent case {record['active_case_id']}")
    tickets = record.get("tickets") or []
    if tickets:
        facts.append(
            "tickets opened by this assistant: "
            + ", ".join(f"{t['ticket_id']} on {t['case_id']} ({t['created_at'][:10]})" for t in tickets[:3])
        )
    if not facts:
        return None
    return (
        "[Application note] Remembered from earlier conversations "
        "(may be out of date): " + "; ".join(facts) + "."
    )


def _load_memory(state: ConversationState, memory: Optional[MemoryStore]) -> None:
    if memory is None or not state.student_id or state.memory_loaded:
        return
    state.memory_loaded = True
    try:
        note = _memory_note(memory.recall(state.student_id))
    except Exception:  # memory is an aid; a failure must not stop the turn
        logger.exception("Could not read memory; continuing without it")
        return
    if note:
        state.history.append({"role": "user", "text": note})


def _status_in_trace(turn_trace: list[dict], case_id: str) -> Optional[str]:
    """The status of this case as returned by get_case_status earlier in this turn."""
    for entry in reversed(turn_trace):
        result = entry["result"]
        if entry["tool"] == "get_case_status" and result.get("success") and result.get("case_id") == case_id:
            return result.get("status")
    return None


def _save_memory(state: ConversationState, memory: Optional[MemoryStore], turn_trace: list[dict]) -> None:
    """Save from tool results only (trusted data), never from model text."""
    if memory is None or not state.student_id:
        return
    try:
        for entry in reversed(turn_trace):
            if entry["tool"] == "create_support_ticket" and entry["result"].get("success"):
                ticket = entry["result"]["ticket"]
                memory.remember_ticket(state.student_id, ticket, _status_in_trace(turn_trace, ticket["case_id"]))
                memory.remember_case(state.student_id, ticket["case_id"])
                return
        for entry in reversed(turn_trace):
            if entry["tool"] == "get_case_status" and entry["result"].get("success"):
                memory.remember_case(state.student_id, entry["result"]["case_id"])
                return
    except Exception:
        logger.exception("Could not save memory")


def _recent_ticket(state: ConversationState, memory: Optional[MemoryStore], case_id: str) -> Optional[dict]:
    """The student's recent ticket on this case, if memory has one. Advisory: failures fail open."""
    if memory is None or not state.student_id:
        return None
    try:
        return memory.find_recent_ticket(state.student_id, case_id)
    except Exception:
        logger.exception("Could not check for a duplicate ticket; continuing without it")
        return None


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

class _AllModelsFailed(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


class _ToolCrashed(Exception):
    pass


def _failure(message: str, blocked: bool = False) -> dict:
    result = {"success": False, "error": message}
    if blocked:
        result["blocked"] = True
    return result


def _record(state: ConversationState, trace: list[dict], tool: str, arguments: dict,
            result: dict, source: str, ignored: Optional[list[str]] = None) -> None:
    entry = {"tool": tool, "arguments": arguments, "result": result, "source": source}
    if ignored:
        entry["ignored_arguments"] = ignored
    trace.append(entry)
    state.observations.append({"tool": tool, "arguments": arguments, "result": result})


def _execute(state: ConversationState, request: ToolRequest, trace: list[dict],
             memory: Optional[MemoryStore] = None) -> dict:
    """Check one model tool request against the contract, run it if allowed, record it."""
    name = request.name
    raw_arguments = request.arguments if isinstance(request.arguments, dict) else {}

    # 1. Allow-list
    if name not in PERMITTED_TOOLS:
        result = _failure(f"The tool '{name}' is not permitted.")
        _record(state, trace, name, raw_arguments, result, "model")
        return result

    schema = SCHEMAS_BY_NAME[name]["parameters"]

    # 2. Arguments must have been valid JSON
    if request.arguments is None:
        result = _failure("The tool arguments were not valid. Ask the student for the details you need.")
        _record(state, trace, name, {}, result, "model")
        return result

    # 3. Drop anything not in the schema (e.g. a student_id from the model)
    allowed = set(schema["properties"])
    arguments = {k: v for k, v in raw_arguments.items() if k in allowed}
    ignored = sorted(set(raw_arguments) - allowed)

    # 4. Missing required arguments: do not call the tool
    missing = [p for p in schema.get("required", []) if p not in arguments]
    if missing:
        result = _failure(
            f"Missing required argument(s): {', '.join(missing)}. "
            "Ask the student for this information; do not guess it."
        )
        _record(state, trace, name, arguments, result, "model", ignored)
        return result

    # 5. Ticket rules enforced from state and case data
    if name == "create_support_ticket":
        if state.ticket_created:
            result = _failure(
                f"A ticket ({state.ticket_id}) has already been created in this "
                "conversation. No further ticket can be created.",
                blocked=True,
            )
            _record(state, trace, name, arguments, result, "model", ignored)
            return result

        # The student must have given this case ID in this conversation. This stops a
        # remembered (or model-invented) ID from opening a ticket on its own.
        case_ref = arguments["case_id"].strip().upper() if isinstance(arguments["case_id"], str) else ""
        if case_ref not in state.student_case_ids:
            result = _failure(
                "The student has not given this case ID in this conversation. Ask the "
                "student to confirm the case ID before a ticket is opened."
            )
            _record(state, trace, name, arguments, result, "model", ignored)
            return result

        # The application looks the case up itself; it does not rely on the model's lookup.
        lookup = TOOL_FUNCTIONS["get_case_status"](arguments["case_id"])
        _record(state, trace, "get_case_status", {"case_id": arguments["case_id"]}, lookup, "application")
        if lookup.get("success") and lookup.get("status") not in TICKET_ALLOWED_STATUSES:
            result = _failure(
                f"Case {lookup.get('case_id')} has status '{lookup.get('status')}'. "
                "A ticket can only be opened on a case that is in progress.",
                blocked=True,
            )
            _record(state, trace, name, arguments, result, "model", ignored)
            return result
        # If the case was not found, the tool itself rejects the request below.

        # Duplicate warning from memory. It holds the ticket back once (and for the rest of
        # that turn); after the student has replied, the ticket may be opened.
        recent = _recent_ticket(state, memory, arguments["case_id"])
        if recent is not None:
            warned_on = state.duplicate_warned_at.setdefault(case_ref, state.turn_count)
            if warned_on == state.turn_count:
                result = _failure(
                    f"The student already opened {recent['ticket_id']} on {case_ref} on "
                    f"{recent['created_at'][:10]}. Tell the student and ask whether they still "
                    "want a new ticket. Open another only if they say yes."
                )
                _record(state, trace, name, arguments, result, "model", ignored)
                return result

    # 6. Run the tool
    result = TOOL_FUNCTIONS[name](**arguments)
    _record(state, trace, name, arguments, result, "model", ignored)

    if isinstance(arguments.get("case_id"), str):
        state.case_id = arguments["case_id"]
    if name == "create_support_ticket" and result.get("success"):
        state.ticket_created = True
        state.ticket_id = result["ticket"]["ticket_id"]
    return result


# ---------------------------------------------------------------------------
# Model calls with fallback
# ---------------------------------------------------------------------------

def _call_model(history: list[dict], providers: list[Any], start: int,
                errors: list[str]) -> tuple[ModelResponse, int]:
    """Try providers in order from `start`. Returns the response and the index that answered."""
    for index in range(start, len(providers)):
        provider = providers[index]
        try:
            response = provider.generate(history)
            if not response.text.strip() and not response.tool_requests:
                raise RuntimeError("empty response")
            return response, index
        except Exception as exc:  # any provider failure moves on to the fallback
            message = f"{provider.name}: {type(exc).__name__}: {exc}"
            logger.warning("Model call failed (%s)", message)
            errors.append(message)
    raise _AllModelsFailed(errors)


def _outcome(turn_trace: list[dict]) -> str:
    """
    Indicative stop condition, derived only from what the application can see
    (tool activity). A reply with no tool calls could be a clarification
    question, a refusal or a direct answer; the application cannot tell which.
    """
    if not turn_trace:
        return "reply_only"
    for entry in turn_trace:
        if entry["tool"] == "create_support_ticket" and entry["result"].get("success"):
            return "goal_reached"
    for entry in turn_trace:
        if entry["tool"] == "get_case_status" and entry["result"].get("status") == "escalated":
            return "handed_off"
    last = turn_trace[-1]["result"]
    if not last.get("success") and not last.get("blocked"):
        return "needs_student_input"
    return "goal_reached"


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def run_turn(message: str, conversation_id: str, student_id: Optional[str] = None) -> dict:
    """
    Run one agent turn.

    Returns:
        {
          "reply": str,                    # what to show the student
          "tool_trace": [                  # every tool call this turn
              {"tool", "arguments", "result", "source", ["ignored_arguments"]}
          ],
          "outcome": str,                  # goal_reached | handed_off | needs_student_input
                                           # | reply_only | round_limit | unrecoverable_failure
          "rounds": int,                   # model calls used this turn (max MAX_TOOL_ROUNDS)
          "model_used": str | None,        # "groq" or "gemini" (last successful call)
          "fallback_used": bool,
          "clarification_asked": bool,     # heuristic ("?" in the reply); for the trace only
          "model_errors": [str],           # diagnostics only; do not show to students
        }
    """
    return _run_turn(message, conversation_id, student_id, PROVIDERS, memory=_default_memory())


_memory_unavailable_logged = False


def _default_memory() -> Optional[MemoryStore]:
    global _memory_unavailable_logged
    try:
        return default_store()
    except Exception as exc:
        if not _memory_unavailable_logged:
            _memory_unavailable_logged = True
            logger.warning("Persistent memory is unavailable; continuing without it (%s)", exc)
        return None


def _run_turn(message: str, conversation_id: str, student_id: Optional[str],
              providers: list[Any], memory: Optional[MemoryStore] = None) -> dict:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message is required.")

    state = _get_state(conversation_id, student_id)

    with state.lock:  # one turn at a time per conversation, so the ticket rule cannot race
        state.turn_count += 1
        _load_memory(state, memory)
        state.student_case_ids |= _case_ids_in(message)
        state.history.append({"role": "user", "text": message})

        turn_trace: list[dict] = []
        model_errors: list[str] = []
        rounds = 0
        provider_index = 0
        fallback_used = False
        model_used: Optional[str] = None
        reply: Optional[str] = None
        outcome: Optional[str] = None

        try:
            while reply is None:
                if rounds >= MAX_TOOL_ROUNDS:
                    reply, outcome = SAFE_ROUND_LIMIT_MESSAGE, "round_limit"
                    break
                rounds += 1

                response, provider_index = _call_model(
                    state.history, providers, provider_index, model_errors
                )
                model_used = providers[provider_index].name
                fallback_used = fallback_used or provider_index > 0

                if not response.tool_requests:
                    reply = response.text.strip()
                    break

                if rounds >= MAX_TOOL_ROUNDS:
                    # No round left to read the results, so do not run the tools.
                    reply, outcome = SAFE_ROUND_LIMIT_MESSAGE, "round_limit"
                    break

                state.history.append(
                    {
                        "role": "assistant",
                        "text": response.text,
                        "tool_calls": [
                            {"id": r.id, "name": r.name, "arguments": r.arguments or {}}
                            for r in response.tool_requests
                        ],
                    }
                )
                for request in response.tool_requests:
                    try:
                        result = _execute(state, request, turn_trace, memory)
                    except Exception as exc:
                        logger.exception("Tool %s failed unexpectedly", request.name)
                        result = _failure("The tool failed unexpectedly.")
                        _record(state, turn_trace, request.name, {}, result, "model")
                        state.history.append(
                            {"role": "tool", "id": request.id, "name": request.name, "result": result}
                        )
                        raise _ToolCrashed(str(exc)) from exc
                    state.history.append(
                        {"role": "tool", "id": request.id, "name": request.name, "result": result}
                    )
        except _AllModelsFailed:
            reply, outcome = SAFE_FAILURE_MESSAGE, "unrecoverable_failure"
        except _ToolCrashed:
            reply, outcome = SAFE_FAILURE_MESSAGE, "unrecoverable_failure"

        if outcome is None:
            outcome = _outcome(turn_trace)
        state.history.append({"role": "assistant", "text": reply})
        _save_memory(state, memory, turn_trace)

        return {
            "reply": reply,
            "tool_trace": turn_trace,
            "outcome": outcome,
            "rounds": rounds,
            "model_used": model_used,
            "fallback_used": fallback_used,
            "clarification_asked": "?" in reply,
            "model_errors": model_errors,
        }