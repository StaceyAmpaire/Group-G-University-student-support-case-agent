"""
Tool schemas for the agent's function calling.

These mirror the 5 approved tools from the Project Charter section 8
(Proposed Tools and Function Contracts). The model can only ever
PROPOSE a call to one of these; the actual execution below is a stub
that will be replaced by real deterministic logic (DB lookups, ticket
creation, etc.) as the RAG/data layer gets built in later weeks.
"""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Retrieve approved university/course information relevant to the student's question, with source provenance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The retrieval query derived from the student's request"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_case_status",
            "description": "Retrieve the status of an existing synthetic support case. Requires student/session authorization.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {"type": "string"},
                    "case_id": {"type": "string"},
                },
                "required": ["student_id", "case_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_student_timetable",
            "description": "Retrieve a synthetic timetable record for the authenticated student.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {"type": "string"},
                },
                "required": ["student_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": "Propose creation of a simulated support ticket. The system validates required fields, category and authorization before actually creating it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": ["registration", "exams", "transcript", "fees", "other"],
                    },
                    "description": {"type": "string"},
                },
                "required": ["student_id", "category", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "route_case",
            "description": "Route a validated case to the correct department/queue using deterministic category-to-queue rules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "category": {"type": "string"},
                },
                "required": ["case_id", "category"],
            },
        },
    },
]


# --- Stub executors -------------------------------------------------
# Replace each of these with real deterministic logic. They exist so
# the agent loop is runnable and demoable before the data layer lands.

def search_knowledge_base(query: str):
    return {"result": f"[stub] would search knowledge base for: {query}", "sources": []}


def get_case_status(student_id: str, case_id: str):
    return {"result": f"[stub] would look up case {case_id} for {student_id}"}


def get_student_timetable(student_id: str):
    return {"result": f"[stub] would look up timetable for {student_id}"}


def create_support_ticket(student_id: str, category: str, description: str):
    return {"result": f"[stub] would validate and create ticket ({category}) for {student_id}"}


def route_case(case_id: str, category: str):
    return {"result": f"[stub] would route case {case_id} to {category} queue"}


TOOL_EXECUTORS = {
    "search_knowledge_base": search_knowledge_base,
    "get_case_status": get_case_status,
    "get_student_timetable": get_student_timetable,
    "create_support_ticket": create_support_ticket,
    "route_case": route_case,
}
