"""
tools.py

Implements the two agent tools for the University Student-Support Case Agent:

1. check_case_status  -> deterministic lookup (Week 1 Boundary Matrix, Row 4)
2. create_support_ticket -> AI proposes, system validates & executes (Row 6)

Neither tool performs a human-approval-gated action. Escalation, closing a
case, or anything touching admissions/grading/fees/discipline is explicitly
out of scope here (Boundary Matrix Rows 8-9) and is NOT implemented.
"""

from datetime import datetime, timezone
import itertools


# ---------------------------------------------------------------------------
# Synthetic case data (stand-in for a real case-management system)
# ---------------------------------------------------------------------------

SYNTHETIC_CASES = {
    "CASE-1045": {
        "case_id": "CASE-1045",
        "student_name": "Grace Nabirye",
        "category": "registration",
        "status": "in_progress",
        "summary": "Course registration not reflecting on student portal.",
        "last_updated": "2026-09-20",
    },
    "CASE-2031": {
        "case_id": "CASE-2031",
        "student_name": "Brian Okello",
        "category": "fees",
        "status": "resolved",
        "summary": "Payment Reference Number (PRN) not recognised by portal.",
        "last_updated": "2026-09-15",
    },
    "CASE-3312": {
        "case_id": "CASE-3312",
        "student_name": "Patricia Ahumuza",
        "category": "it_support",
        "status": "escalated",
        "summary": "Email conflict preventing access to student webmail.",
        "last_updated": "2026-09-22",
    },
}


# ---------------------------------------------------------------------------
# Simulated ticket store
# ---------------------------------------------------------------------------

_ticket_id_counter = itertools.count(1001)
TICKET_STORE = {}

# Categories the system recognises. The model may suggest a category, but
# only one from this fixed list is ever accepted (deterministic validation,
# per Boundary Matrix Row 7 — the AI cannot invent a routing destination).
ALLOWED_TICKET_CATEGORIES = {
    "registration",
    "fees",
    "it_support",
    "accommodation",
    "academic_records",
    "other",
}


# ---------------------------------------------------------------------------
# Tool 1: Check Case Status
# ---------------------------------------------------------------------------

def check_case_status(case_id: str) -> dict:
    """
    Deterministic lookup against synthetic case data.

    Purpose: let the agent answer "what's the status of my case?" questions
    without the model inventing an answer.

    Input schema: { "case_id": string, required }
    Output schema (success):
        { "found": true, "case_id", "student_name", "category",
          "status", "summary", "last_updated" }
    Output schema (not found):
        { "found": false, "case_id", "message" }
    Failure behaviour: never raises — always returns a structured dict so
    the model has something well-formed to reason over, even for an unknown
    or malformed case ID.
    """
    if not case_id or not isinstance(case_id, str):
        return {
            "found": False,
            "case_id": case_id,
            "message": "No case ID was provided. Please supply a case ID, e.g. CASE-1045.",
        }

    case = SYNTHETIC_CASES.get(case_id.strip().upper())

    if case is None:
        return {
            "found": False,
            "case_id": case_id,
            "message": f"No case was found with ID '{case_id}'. Please check the ID and try again.",
        }

    return {"found": True, **case}


# ---------------------------------------------------------------------------
# Tool 2: Create Support Ticket
# ---------------------------------------------------------------------------

def create_support_ticket(student_name: str, category: str, description: str) -> dict:
    """
    AI proposes ticket content; this function deterministically validates
    and executes the write (Boundary Matrix Row 6).

    Purpose: let the agent turn an unresolved student issue into a tracked
    ticket instead of leaving it unanswered.

    Input schema:
        { "student_name": string, required,
          "category": string, required, must be one of ALLOWED_TICKET_CATEGORIES,
          "description": string, required }
    Output schema (success):
        { "created": true, "ticket_id", "student_name", "category",
          "description", "status", "created_at" }
    Output schema (validation failure):
        { "created": false, "errors": [string, ...] }
    Failure behaviour: never raises — missing or invalid fields are
    collected into "errors" and returned so the agent can ask the student
    for the missing information rather than crash or silently guess.
    """
    errors = []

    if not student_name or not str(student_name).strip():
        errors.append("student_name is required.")

    if not category or not str(category).strip():
        errors.append("category is required.")
    elif category.strip().lower() not in ALLOWED_TICKET_CATEGORIES:
        errors.append(
            f"'{category}' is not a recognised category. "
            f"Allowed categories: {', '.join(sorted(ALLOWED_TICKET_CATEGORIES))}."
        )

    if not description or not str(description).strip():
        errors.append("description is required.")

    if errors:
        return {"created": False, "errors": errors}

    ticket_id = f"TICKET-{next(_ticket_id_counter)}"
    ticket = {
        "ticket_id": ticket_id,
        "student_name": student_name.strip(),
        "category": category.strip().lower(),
        "description": description.strip(),
        "status": "open",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    TICKET_STORE[ticket_id] = ticket

    return {"created": True, **ticket}


# ---------------------------------------------------------------------------
# Gemini function-calling schemas
# ---------------------------------------------------------------------------
# Declared as plain dicts here so main.py can wrap them in
# google.genai.types.FunctionDeclaration without this module depending on
# the genai SDK directly.

CHECK_CASE_STATUS_SCHEMA = {
    "name": "check_case_status",
    "description": (
        "Look up the status and details of an existing student support case "
        "using its case ID. Use this when a student asks about the status, "
        "progress, or details of a case they already have."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "case_id": {
                "type": "string",
                "description": "The case ID, e.g. CASE-1045.",
            },
        },
        "required": ["case_id"],
    },
}

CREATE_SUPPORT_TICKET_SCHEMA = {
    "name": "create_support_ticket",
    "description": (
        "Create a new support ticket for a student issue that is not already "
        "resolved by retrieved knowledge or an existing case. Use this when "
        "a student reports a new problem that needs human follow-up."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "student_name": {
                "type": "string",
                "description": "The student's full name.",
            },
            "category": {
                "type": "string",
                "description": (
                    "Best-guess category for the issue. Must be one of: "
                    + ", ".join(sorted(ALLOWED_TICKET_CATEGORIES))
                ),
            },
            "description": {
                "type": "string",
                "description": "A clear description of the student's issue.",
            },
        },
        "required": ["student_name", "category", "description"],
    },
}

TOOL_SCHEMAS = [CHECK_CASE_STATUS_SCHEMA, CREATE_SUPPORT_TICKET_SCHEMA]

# Dispatch table used by main.py to execute a tool call by name
TOOL_FUNCTIONS = {
    "check_case_status": check_case_status,
    "create_support_ticket": create_support_ticket,
}
