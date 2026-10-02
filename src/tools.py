"""
tools.py

Implements the two authorized tools for the
University Student-Support Case Agent.

Tools:
1. get_case_status
   - Deterministic lookup of an existing synthetic support case.

2. create_support_ticket
   - Creates a low-risk simulated support ticket.
   - The application validates the request before creating it.

The model can REQUEST these tools, but it cannot execute arbitrary
functions. Tool execution is controlled by the application layer
in main.py.

Out of scope:
- Admissions decisions
- Grading decisions
- Fee decisions
- Disciplinary decisions
- Escalating cases
- Closing/resolving cases
- Changing university records
- Any human-approval-gated action
"""

from datetime import datetime, timezone
import itertools


# ============================================================
# SYNTHETIC CASE DATA
# ============================================================

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


# ============================================================
# SIMULATED TICKET STORE
# ============================================================

_ticket_id_counter = itertools.count(1001)

TICKET_STORE = {}


# ============================================================
# ALLOWED TICKET CATEGORIES
# ============================================================

ALLOWED_TICKET_CATEGORIES = {
    "registration",
    "fees",
    "it_support",
    "accommodation",
    "academic_records",
    "other",
}


# ============================================================
# TOOL 1: GET CASE STATUS
# ============================================================

def get_case_status(case_id: str) -> dict:
    """
    Deterministically retrieve an existing synthetic case.

    Input:
        case_id: existing case identifier, e.g. CASE-1045

    Success output:
        {
            "success": True,
            "case_id": ...,
            "student_name": ...,
            "category": ...,
            "status": ...,
            "summary": ...,
            "last_updated": ...
        }

    Failure output:
        {
            "success": False,
            "case_id": ...,
            "error": ...
        }

    The function does not invent information and does not raise
    exceptions for normal invalid input.
    """

    if not isinstance(case_id, str) or not case_id.strip():
        return {
            "success": False,
            "case_id": case_id,
            "error": "case_id is required.",
        }

    normalized_case_id = case_id.strip().upper()

    case = SYNTHETIC_CASES.get(normalized_case_id)

    if case is None:
        return {
            "success": False,
            "case_id": normalized_case_id,
            "error": f"No case found for {normalized_case_id}.",
        }

    return {
        "success": True,
        **case,
    }


# ============================================================
# TOOL 2: CREATE SUPPORT TICKET
# ============================================================

def create_support_ticket(
    case_id: str,
    category: str,
    description: str,
) -> dict:
    """
    Create a low-risk simulated support ticket.

    The application validates:
    - case ID exists
    - category is provided
    - category is allowed
    - description is provided

    This function does NOT:
    - change university records
    - change grades
    - change fees
    - escalate a case
    - close a case
    - make an admissions or disciplinary decision

    Input:
        case_id: existing support case ID
        category: approved ticket category
        description: student's issue

    Success output:
        {
            "success": True,
            "ticket": {...}
        }

    Validation failure:
        {
            "success": False,
            "errors": [...]
        }
    """

    errors = []

    # Validate case ID
    if not isinstance(case_id, str) or not case_id.strip():
        errors.append("case_id is required.")
    else:
        case_id = case_id.strip().upper()

        if case_id not in SYNTHETIC_CASES:
            errors.append(
                f"No existing case was found for {case_id}."
            )

    # Validate category
    if not isinstance(category, str) or not category.strip():
        errors.append("category is required.")
    else:
        category = category.strip().lower()

        if category not in ALLOWED_TICKET_CATEGORIES:
            errors.append(
                f"'{category}' is not an allowed category. "
                f"Allowed categories: "
                f"{', '.join(sorted(ALLOWED_TICKET_CATEGORIES))}."
            )

    # Validate description
    if not isinstance(description, str) or not description.strip():
        errors.append("description is required.")
    else:
        description = description.strip()

    # Do not create anything if validation failed
    if errors:
        return {
            "success": False,
            "errors": errors,
        }

    # Create simulated ticket
    ticket_id = f"TICKET-{next(_ticket_id_counter)}"

    ticket = {
        "ticket_id": ticket_id,
        "case_id": case_id,
        "category": category,
        "description": description,
        "status": "open",
        "created_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
    }

    TICKET_STORE[ticket_id] = ticket

    return {
        "success": True,
        "ticket": ticket,
    }


# ============================================================
# TOOL SCHEMAS
# ============================================================

GET_CASE_STATUS_SCHEMA = {
    "name": "get_case_status",
    "description": (
        "Retrieve the current status and details of an existing "
        "student support case using its case ID."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "case_id": {
                "type": "string",
                "description": (
                    "The existing student support case ID, "
                    "for example CASE-1045."
                ),
            }
        },
        "required": ["case_id"],
    },
}


CREATE_SUPPORT_TICKET_SCHEMA = {
    "name": "create_support_ticket",
    "description": (
        "Create a low-risk simulated support ticket for an "
        "existing student support case. The application validates "
        "the case ID and ticket category before creating the ticket."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "case_id": {
                "type": "string",
                "description": (
                    "An existing student support case ID."
                ),
            },
            "category": {
                "type": "string",
                "description": (
                    "The support category. Must be one of: "
                    + ", ".join(sorted(ALLOWED_TICKET_CATEGORIES))
                ),
            },
            "description": {
                "type": "string",
                "description": (
                    "A clear description of the student's issue."
                ),
            },
        },
        "required": [
            "case_id",
            "category",
            "description",
        ],
    },
}


TOOL_SCHEMAS = [
    GET_CASE_STATUS_SCHEMA,
    CREATE_SUPPORT_TICKET_SCHEMA,
]


# ============================================================
# AUTHORIZED TOOL DISPATCH TABLE
# ============================================================

TOOL_FUNCTIONS = {
    "get_case_status": get_case_status,
    "create_support_ticket": create_support_ticket,
}