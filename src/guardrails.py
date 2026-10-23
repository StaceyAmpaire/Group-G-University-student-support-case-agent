"""
guardrails.py  (Week 7)

Application-side guardrails for the Student-Support Case Agent.

Why these live in the application and not in the prompt
    A prompt is a request; the model can ignore it, or be talked out of it by a
    student or by text hidden in a tool result. The scripted evaluation showed this:
    "gullible" models that did what an attacker asked got through, because nothing
    between the model and the student checked what came out. These checks run in
    code, whatever the model does.

Three layers (each is a pure function, so each can be unit-tested on its own)
    1. screen_input(message, student_id, require_student_id)   before any model call
         - refuses a request with no signed-in student (fail closed)
         - rejects empty, oversized or non-text input (the model is never called)
         - strips control and invisible characters
         - refuses requests to reveal the system instructions
         - flags (does not block) "ignore your rules" style phrasing for the trace
    2. sanitize_tool_result(...)     before a tool result reaches the model
         - drops fields the model does not need (the student's name)
         - validates structured fields against their expected format
         - withholds free text that reads like an instruction to the assistant
    3. screen_output(...)            before a reply reaches the student
         - claims of actions the agent cannot take (grades, closing, escalating)
         - claims that a ticket was created when none was
         - case statuses that do not match what the tools returned
         - ticket IDs, case IDs or other reference codes the application never produced
         - promises and timeframes, money amounts and dates nobody supplied
         - leaks of the system instructions
         - repeats of text that was withheld from a tool result as an injection
       A reply that fails is replaced by a short grounded message built only from
       application data (safe_response). It never repeats the blocked text.

Every function reports what it did, so bounded_agent.py can record a trace event
(codes only, never student content) for each guardrail that fires.

Changes after the first review (all covered by test_guardrails.py)
    - screen_input now refuses an unauthenticated request (UNA-04).
    - Prompt-extraction patterns need an extraction cue, so "my original message" or
      "a system message on the portal" are no longer refused.
    - An example ID given in a clarifying question ("for example CASE-1234") is not
      treated as an invented reference.
    - Added: ticket claims with no ticket, "I updated ...", "has been updated/changed",
      and negation handling ("I have not updated" is not a claim).
    - Date matching uses real month names, so "marked 3 times" is not a date.
    - Added the withheld-text echo check (injection_echo).
UNCONFIRMED: MAX_INPUT_CHARS must be checked against the Week 7 brief / harness.
"""

from __future__ import annotations

import copy
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

MAX_INPUT_CHARS = 1000

WITHHELD_TEXT = "[text withheld: it contained instruction-like content]"
UNAVAILABLE_TEXT = "[unavailable]"


# ---------------------------------------------------------------------------
# 1. Input screening
# ---------------------------------------------------------------------------

EMPTY_REPLY = "I didn't receive a message. Please tell me how I can help with a support case."
TOO_LONG_REPLY = (
    f"Your message is too long for me to process (the limit is {MAX_INPUT_CHARS} characters). "
    "Please send a shorter message with your case ID and a brief description."
)
INVALID_REPLY = "I couldn't read that message. Please send your question as plain text."
EXTRACTION_REPLY = (
    "I can't share my internal instructions. I can check the status of a case or help you "
    "follow up on one."
)
UNAUTHENTICATED_REPLY = (
    "I can't look up or change a case until you are signed in as a student. "
    "Please sign in and try again."
)
TOOL_FAILURE_REPLY = (
    "I'm sorry, I couldn't complete that because a system I rely on is unavailable. "
    "Please try again in a few minutes."
)

_VERBS = r"(?:print|show|reveal|repeat|display|output|recite|leak|dump|share|tell me|give me|list|what(?:'s| is| are))"
_PROMPT_EXTRACTION = [
    # "what is your system prompt", "your hidden instructions"
    re.compile(r"\byour\s+(?:system|hidden|internal|initial|original|secret)\s+(?:prompt|instructions?|rules?)\b", re.I),
    # "print the system prompt / hidden instructions"
    re.compile(rf"\b{_VERBS}\b.{{0,40}}\b(?:system|hidden|internal|secret)\s+(?:prompt|instructions?)\b", re.I),
    # "print your instructions / rules / tool list"
    re.compile(
        rf"\b{_VERBS}\b.{{0,40}}\byour\s+(?:instructions?|prompt|rules|guidelines|tool list|tools list|configuration)\b",
        re.I,
    ),
    # "repeat the initial instructions you were given"
    re.compile(r"\b(?:initial|original)\s+(?:prompt|instructions?)\b.{0,30}\b(?:you were given|you received|given to you)\b", re.I),
]

_INJECTION_PHRASES = [
    re.compile(r"\bignore\b.{0,30}\b(?:instructions?|rules?|previous|prior|above)\b", re.I),
    re.compile(r"\b(?:disregard|forget|override)\b.{0,30}\b(?:instructions?|rules?|previous|prior|above)\b", re.I),
    re.compile(r"\byou are now\b|\bnew instructions?\b|\bfrom now on you\b", re.I),
]


@dataclass
class InputDecision:
    ok: bool
    message: str                      # the cleaned message (use this, not the original)
    code: Optional[str] = None        # why it was rejected
    reply: Optional[str] = None       # what to tell the student if rejected
    flags: list = field(default_factory=list)   # things noticed but not blocked


def _clean(text: str) -> str:
    """Remove control and invisible formatting characters, keeping newlines and tabs."""
    return "".join(
        ch for ch in text
        if ch in "\n\t" or unicodedata.category(ch) not in ("Cc", "Cf")
    )


def screen_input(message: Any, student_id: Any = None, require_student_id: bool = False) -> InputDecision:
    """
    require_student_id=True refuses the request when no signed-in student is supplied.
    The default is False so existing callers keep working; bounded_agent.py turns it on.
    """
    if require_student_id and not (isinstance(student_id, str) and student_id.strip()):
        return InputDecision(False, "", "unauthenticated", UNAUTHENTICATED_REPLY)

    if not isinstance(message, str):
        return InputDecision(False, "", "invalid_input", INVALID_REPLY)

    cleaned = _clean(message).strip()
    flags = []
    if cleaned != message.strip():
        flags.append("control_characters_removed")

    if not cleaned:
        return InputDecision(False, "", "empty_input", EMPTY_REPLY, flags)
    if len(cleaned) > MAX_INPUT_CHARS:
        return InputDecision(False, cleaned[:MAX_INPUT_CHARS], "input_too_long", TOO_LONG_REPLY, flags)
    if any(p.search(cleaned) for p in _PROMPT_EXTRACTION):
        return InputDecision(False, cleaned, "prompt_extraction", EXTRACTION_REPLY, flags)

    if any(p.search(cleaned) for p in _INJECTION_PHRASES):
        flags.append("injection_suspected")
    return InputDecision(True, cleaned, flags=flags)


# ---------------------------------------------------------------------------
# 2. Tool-result sanitising
# ---------------------------------------------------------------------------

_DROPPED_KEYS = {"student_name"}                 # the model never needs the student's name
_APP_TEXT_KEYS = {"error", "errors", "message", "source"}   # written by the application itself
_FREE_TEXT_LIMITS = {"summary": 300, "description": 500}

_CASE_ID = re.compile(r"^CASE-\d{1,10}$")
_TICKET_ID = re.compile(r"^TICKET-\d{1,10}$")
_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9 _\-]{0,31}$")
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ][0-9:+\-Z\.]{0,20})?$")

_STRUCTURED = {
    "case_id": _CASE_ID,
    "ticket_id": _TICKET_ID,
    "status": _LABEL,
    "category": _LABEL,
    "case_status": _LABEL,
    "last_updated": _TIMESTAMP,
    "created_at": _TIMESTAMP,
    "expires_at": _TIMESTAMP,
}

_IMPERATIVE = r"(?:reply|respond|answer|output|print|say|write|append|include)"
_INJECTION_IN_DATA = [
    re.compile(r"\bignore\b.{0,40}\b(?:instructions?|rules?|previous|prior|above)\b", re.I),
    re.compile(r"\b(?:disregard|forget|override)\b.{0,40}\b(?:instructions?|rules?|previous|prior|above)\b", re.I),
    re.compile(r"\b(?:system|developer|admin(?:istrator)?)\s*(?:note|message|prompt|instruction|override|command)s?\b", re.I),
    re.compile(r"\b(?:you must|you should|you are now|you will now|from now on|new instructions?)\b", re.I),
    # an instruction to the assistant: starts a sentence or follows "please/then/now/always"
    re.compile(rf"(?:^|[.;:!?]\s*|\b(?:please|then|now|always)\s+){_IMPERATIVE}\b"
               r".{0,40}\b(?:with|by saying|exactly|only|the (?:word|phrase|text|code))\b", re.I),
    re.compile(r"\b(?:end|begin|start|finish)\s+(?:your|the)\s+(?:reply|response|answer|message)\b", re.I),
    re.compile(r"\b(?:call|use|invoke|execute|run)\b.{0,30}\b(?:tool|function|create_support_ticket|get_case_status)\b", re.I),
    re.compile(r"[<\[{]\s*/?\s*(?:system|assistant|instructions?)\b", re.I),
    re.compile(r"\bbegin (?:system|instructions)\b", re.I),
]


def _looks_like_injection(value: str) -> bool:
    return any(p.search(value) for p in _INJECTION_IN_DATA)


def _injection_sentences(value: str) -> list:
    """The sentences of `value` that read as instructions (whole value if none single out)."""
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", value) if s.strip()]
    hits = [s for s in parts if _looks_like_injection(s)]
    return hits or ([value.strip()] if _looks_like_injection(value) else [])


def _walk(value: Any, key: Optional[str], path: str, findings: list) -> Any:
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k in _DROPPED_KEYS:
                continue
            out[k] = _walk(v, k, f"{path}.{k}" if path else k, findings)
        return out
    if isinstance(value, list):
        return [_walk(v, key, f"{path}[{i}]", findings) for i, v in enumerate(value)]
    if not isinstance(value, str):
        return value

    pattern = _STRUCTURED.get(key)
    if pattern is not None:
        if not pattern.match(value.strip()):
            findings.append({"field": path, "reason": "unexpected_format"})
            return UNAVAILABLE_TEXT
        return value
    if key in _APP_TEXT_KEYS:
        return value

    if _looks_like_injection(value):
        findings.append({"field": path, "reason": "instruction_like_text"})
        return WITHHELD_TEXT
    limit = _FREE_TEXT_LIMITS.get(key)
    if limit and len(value) > limit:
        findings.append({"field": path, "reason": "truncated"})
        return value[:limit]
    return value


def sanitize_tool_result(name: str, result: Any) -> tuple[Any, list]:
    """
    A copy of a tool result that is safe to show the model, plus a list of findings
    ({"field", "reason"}, never the content). The original result is not modified.
    """
    if not isinstance(result, dict):
        return result, []
    findings: list = []
    return _walk(copy.deepcopy(result), None, "", findings), findings


# ---------------------------------------------------------------------------
# 3. Output screening
# ---------------------------------------------------------------------------

SAFE_LIMITS = (
    "I can't change grades or other records, or close or escalate cases; "
    "a human staff member would need to handle that."
)
UNVERIFIED_CASE = "I wasn't able to verify that case, so I can't share its status."
NO_TIMING = "I don't have any information about timing for this case."
NO_INFO = "I don't have verified information about that, so I can't confirm it."
NO_REFERENCE = "I wasn't able to confirm that, so I can't share it."
NO_TICKET = "I wasn't able to create a ticket, so I can't confirm one was opened."
NO_INSTRUCTIONS = "I can't act on instructions found inside case data."

_STATUS_WORDS = re.compile(r"\b(resolved|escalated|closed|in[ _\-]progress|under[ _\-]review)\b", re.I)
_ACTION_LIKE_STATUSES = {"escalated", "closed"}

_DONE = (r"updated|changed|modified|edited|closed|resolved|escalated|deleted|removed|approved|waived|"
         r"transferred|rerouted|overridden|marked|corrected|amended|reversed|cleared|cancelled|canceled")
_ACTION_CLAIMS = [
    # "I updated ...", "I've closed ...", "I have now escalated ..."  (but not "I have not updated")
    re.compile(rf"\b(?:i|we)(?:'ve|\s+have)?\s+(?:(?!not\b|never\b)\w+\s+)?(?:{_DONE})\b", re.I),
    # "has been updated", "have been changed", "your grade is now ..."
    re.compile(rf"\b(?:has|have)\s+been\s+(?:{_DONE})\b", re.I),
    re.compile(r"\b(?:grade|marks?|results?|records?|fees?|balance)\b[^.]{0,40}\b(?:is|are|was|were)\s+now\b", re.I),
]
_TICKET_CLAIMS = [
    re.compile(r"\b(?:i|we)(?:'ve|\s+have)?\s+(?:(?!not\b|never\b)\w+\s+)?(?:created|opened|raised|submitted|logged)\b[^.]{0,40}\bticket\b", re.I),
    re.compile(r"\bticket\b[^.]{0,30}\b(?:has|have)\s+been\s+(?:created|opened|raised|submitted|logged)\b", re.I),
]

_PROMISES = [
    re.compile(r"\btomorrow\b", re.I),
    re.compile(r"\bwithin\s+(?:the next\s+)?\d+\s*(?:minutes?|hours?|days?|weeks?|working days?)\b", re.I),
    re.compile(r"\bwill be\s+(?:fixed|resolved|completed|sorted|processed|done)\b", re.I),
    re.compile(r"\b(?:will|'ll|going to)\s+(?:contact|call|email|get back to|reach out to|respond to|follow up with)\s+you\b", re.I),
    re.compile(r"\bby\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|the end of (?:the )?(?:day|week))\b", re.I),
]

_MONEY = [
    re.compile(r"\b(?:ugx|ush|shs?|usd)\s?\d", re.I),
    re.compile(r"\$\s?\d"),
    re.compile(r"\b\d[\d,\.]*\s?(?:ugx|ush|shs|usd)\b", re.I),
    re.compile(r"\b\d{1,3}(?:,\d{3})+\b"),
]

_MONTHS = {}
for _i, _names in enumerate([
    ("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"), ("may",),
    ("jun", "june"), ("jul", "july"), ("aug", "august"), ("sep", "sept", "september"),
    ("oct", "october"), ("nov", "november"), ("dec", "december"),
], start=1):
    for _n in _names:
        _MONTHS[_n] = _i
_DAY_MONTH = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([A-Za-z]{3,9})\b")
_MONTH_DAY = re.compile(r"\b([A-Za-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?\b")
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

_CODE = re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+\b")
_EXAMPLE_CODE = re.compile(
    r"(?:for example|for instance|e\.g\.?|such as)[:,]?\s*\(?\s*([A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+)", re.I)


def _normalise_status(word: str) -> str:
    return re.sub(r"[ _\-]+", " ", word.strip().lower())


def _dates_in(text: str) -> set:
    found = set()
    for year, month, day in _ISO_DATE.findall(text):
        found.add((int(month), int(day)))
    for day, month in _DAY_MONTH.findall(text):
        if month.lower() in _MONTHS and 1 <= int(day) <= 31:
            found.add((_MONTHS[month.lower()], int(day)))
    for month, day in _MONTH_DAY.findall(text):
        if month.lower() in _MONTHS and 1 <= int(day) <= 31:
            found.add((_MONTHS[month.lower()], int(day)))
    return found


def _words(text: str) -> list:
    return re.sub(r"[^a-z0-9_ ]+", " ", text.lower()).split()


def _shares_run(text: str, source: str, n: int) -> bool:
    """True if `text` repeats n or more consecutive words of `source`."""
    source_words, text_words = _words(source), _words(text)
    grams = {tuple(source_words[i:i + n]) for i in range(len(source_words) - n + 1)}
    return any(tuple(text_words[i:i + n]) in grams for i in range(len(text_words) - n + 1))


_TRIGGER = re.compile(r"\b(?:say|saying|reply with|respond with|answer with|output|print|write|append|include|exactly|only)\b", re.I)
_FILLER = {"the", "a", "an", "text", "phrase", "word", "words", "code", "string", "exactly", "only", "following",
           "that", "to", "me", "your", "reply", "response", "answer", "message"}


def _payload(fragment: str) -> str:
    """What an injected instruction asked the model to say ('' if it cannot be told)."""
    parts = _TRIGGER.split(fragment)
    if len(parts) < 2:
        return ""
    words = [w for w in _words(parts[-1]) if w not in _FILLER]
    return " ".join(words) if len(words) >= 2 else ""


@dataclass
class Facts:
    """What the application actually knows after a run. Output is checked against this only."""
    student_message: str = ""
    system_instruction: str = ""
    statuses: set = field(default_factory=set)          # normalised statuses from successful lookups
    case_ids: set = field(default_factory=set)
    ticket_ids: set = field(default_factory=set)        # created this run, or remembered from earlier
    known_dates: set = field(default_factory=set)       # (month, day) from structured date fields
    lookup: Optional[dict] = None                       # the latest successful lookup
    ticket_created: bool = False
    withheld: list = field(default_factory=list)        # instruction-like text removed from tool data


def build_facts(student_message: str, system_instruction: str, tool_results: Iterable[dict]) -> Facts:
    """Collect facts from the RAW tool results of this run (never from model text)."""
    facts = Facts(student_message=student_message or "", system_instruction=system_instruction or "")

    def collect(value: Any, key: Optional[str] = None) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                collect(v, k)
        elif isinstance(value, list):
            for v in value:
                collect(v, key)
        elif isinstance(value, str):
            if key == "case_id" and _CASE_ID.match(value.strip().upper()):
                facts.case_ids.add(value.strip().upper())
            elif key == "ticket_id" and _TICKET_ID.match(value.strip().upper()):
                facts.ticket_ids.add(value.strip().upper())
            elif key in ("last_updated", "created_at"):
                facts.known_dates |= _dates_in(value)
            elif key not in _STRUCTURED and key not in _APP_TEXT_KEYS and key not in _DROPPED_KEYS:
                facts.withheld.extend(_injection_sentences(value))

    for result in tool_results:
        if not isinstance(result, dict):
            continue
        collect(result)
        if result.get("success") and result.get("status"):
            facts.statuses.add(_normalise_status(str(result["status"])))
            facts.lookup = result
        ticket = result.get("ticket")
        if result.get("success") and isinstance(ticket, dict) and ticket.get("ticket_id"):
            facts.ticket_created = True
    return facts


def _prompt_leak(text: str, system_instruction: str) -> bool:
    if not system_instruction:
        return False
    # Section headings such as "AVAILABLE TOOLS:" copied from the prompt.
    for line in system_instruction.splitlines():
        heading = line.strip()
        if re.fullmatch(r"[A-Z][A-Z ]{3,}:", heading) and heading[:-1] in text:
            return True
    # A long verbatim run of the prompt (10+ consecutive words).
    return _shares_run(text, system_instruction, 10)


def screen_output(text: str, facts: Facts) -> list:
    """Return the guardrail codes a reply violates (empty list means it is fine)."""
    violations = []
    if not isinstance(text, str) or not text.strip():
        return violations

    if _prompt_leak(text, facts.system_instruction):
        violations.append("prompt_leak")

    if any(p.search(text) for p in _ACTION_CLAIMS):
        violations.append("action_claim")

    if not facts.ticket_created and any(p.search(text) for p in _TICKET_CLAIMS):
        violations.append("unsupported_ticket_claim")

    claimed = {_normalise_status(m) for m in _STATUS_WORDS.findall(text)}
    if claimed - facts.statuses:
        violations.append("status_mismatch")

    allowed_codes = {c.upper() for c in _CODE.findall(facts.student_message.upper())}
    allowed_codes |= facts.case_ids | facts.ticket_ids
    allowed_codes |= {c.upper() for c in _EXAMPLE_CODE.findall(text)}      # "for example CASE-1234"
    if any(code not in allowed_codes for code in _CODE.findall(text)):
        violations.append("invented_reference")

    if facts.withheld:
        known = facts.case_ids | facts.ticket_ids
        for fragment in facts.withheld:
            codes = {c for c in _CODE.findall(fragment) if c not in known}
            payload = _payload(fragment)
            if (any(c in text for c in codes) or _shares_run(text, fragment, 6)
                    or (payload and f" {payload} " in f" {' '.join(_words(text))} ")):
                violations.append("injection_echo")
                break

    if any(p.search(text) for p in _PROMISES):
        violations.append("unsupported_promise")

    if any(p.search(text) for p in _MONEY) or (_dates_in(text) - facts.known_dates):
        violations.append("ungrounded_fact")

    return violations


def _status_line(facts: Facts) -> Optional[str]:
    if not facts.lookup:
        return None
    status = _normalise_status(str(facts.lookup.get("status", "")))
    line = f"{facts.lookup.get('case_id', 'The case')} is recorded as {status}"
    updated = str(facts.lookup.get("last_updated", "")).strip()
    return f"{line} (last updated {updated})." if updated else f"{line}."


def safe_response(violations: list, facts: Facts, original: str = "") -> str:
    """
    A short reply built only from application data. It never repeats the blocked text.
    """
    codes = set(violations)
    if "prompt_leak" in codes:
        return EXTRACTION_REPLY

    grounded = _status_line(facts)
    parts: list = []

    claimed = {_normalise_status(m) for m in _STATUS_WORDS.findall(original or "")}
    action_status_claim = bool(claimed & _ACTION_LIKE_STATUSES) and "status_mismatch" in codes

    if "injection_echo" in codes:
        parts.append(NO_INSTRUCTIONS)
        if grounded:
            parts.append(grounded)

    if "action_claim" in codes or action_status_claim:
        if grounded:
            parts.append(grounded)
        elif "status_mismatch" in codes:
            parts.append(UNVERIFIED_CASE)
        parts.append(SAFE_LIMITS)
    elif "status_mismatch" in codes:
        parts.append(grounded or UNVERIFIED_CASE)

    if "unsupported_ticket_claim" in codes:
        parts.append(NO_TICKET)
    if "unsupported_promise" in codes:
        parts.append(NO_TIMING)
    if "ungrounded_fact" in codes:
        parts.append(NO_INFO)
    if "invented_reference" in codes and not parts:
        if grounded:
            parts.append(grounded)
        elif re.search(r"TICKET-\d+", original or "") and not facts.ticket_created:
            parts.append(NO_TICKET)
        else:
            parts.append(NO_REFERENCE)

    seen, unique = set(), []
    for part in parts:
        if part not in seen:
            seen.add(part)
            unique.append(part)
    return " ".join(unique) if unique else NO_REFERENCE
