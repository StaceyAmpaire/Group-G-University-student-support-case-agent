"""
mcp_interface.py  (Week 6)

MCP-style interface for the Student-Support Case capability.

This is an in-process, dependency-free implementation of the *shape* of the
Model Context Protocol: JSON-RPC 2.0 messages with `initialize`, `tools/list`
and `tools/call`, each tool described by a name, description and JSON input
schema. It is not a networked MCP server; the same dispatcher could be placed
behind the official MCP SDK's stdio/HTTP transport without changing the tools.

SECURITY BOUNDARY (enforced here, not by the model or the caller):
  1. Authentication: the caller identity (student_id + scopes) is supplied by the
     host application, never read from tool arguments.
  2. Authorisation: every tool declares a required scope; a caller lacking it
     gets error -32001.
  3. Ownership: students can only see/act on their own cases and memory.
     Another student's case looks exactly like "not found".
  4. Input validation: arguments are checked against the tool's schema
     (required fields, types, enums, no unknown fields, length caps).
  5. Output minimisation: student_name is never returned over this interface.
  6. Human approval: `ticket.create` requires confirmed_by_student=true, which
     the host sets only after the student explicitly confirms.
  7. Audit: every call is logged (tool, outcome, student_ref) without content.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from tools import get_case_status, create_support_ticket, CASE_OWNERS, ALLOWED_TICKET_CATEGORIES
from state import student_ref
from memory import MemoryStore

PROTOCOL_VERSION = "2026-10-draft-style"   # MCP-style, not the official spec version
SERVER_INFO = {"name": "student-support-case-server", "version": "0.6.0"}

# JSON-RPC / custom error codes
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS = -32700, -32600, -32601, -32602
FORBIDDEN, NOT_ALLOWED, TOOL_FAILED = -32001, -32002, -32003

SCOPES = {"cases:read", "tickets:write", "memory:read", "memory:delete"}


@dataclass
class Caller:
    """Authenticated identity supplied by the host application."""
    student_id: str
    scopes: set = field(default_factory=set)


TOOL_DEFINITIONS = [
    {
        "name": "case.get_status",
        "description": "Read the status of one of the caller's own support cases.",
        "required_scope": "cases:read",
        "side_effects": "none (read-only)",
        "inputSchema": {
            "type": "object",
            "properties": {"case_id": {"type": "string", "pattern": r"^CASE-\d{4}$", "maxLength": 12}},
            "required": ["case_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ticket.create",
        "description": ("Create a simulated follow-up ticket for the caller's own case "
                        "that is in progress or under review. Requires explicit student confirmation."),
        "required_scope": "tickets:write",
        "side_effects": "creates a simulated ticket record",
        "inputSchema": {
            "type": "object",
            "properties": {
                "case_id": {"type": "string", "pattern": r"^CASE-\d{4}$", "maxLength": 12},
                "category": {"type": "string", "enum": sorted(ALLOWED_TICKET_CATEGORIES)},
                "description": {"type": "string", "minLength": 5, "maxLength": 500},
                "confirmed_by_student": {"type": "boolean"},
            },
            "required": ["case_id", "category", "description", "confirmed_by_student"],
            "additionalProperties": False,
        },
    },
    {
        "name": "memory.get_history",
        "description": "Return the caller's own saved ticket history (right of access).",
        "required_scope": "memory:read",
        "side_effects": "none (read-only; access is audit-logged)",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "memory.forget",
        "description": "Delete all of the caller's saved ticket history.",
        "required_scope": "memory:delete",
        "side_effects": "deletes the caller's memory records",
        "inputSchema": {
            "type": "object",
            "properties": {"confirm": {"type": "boolean"}},
            "required": ["confirm"],
            "additionalProperties": False,
        },
    },
]
_TOOLS = {t["name"]: t for t in TOOL_DEFINITIONS}

_TYPES = {"string": str, "boolean": bool, "object": dict}


def validate_arguments(schema: dict, args: Any) -> Optional[str]:
    """Minimal JSON-Schema subset validator. Returns an error string or None."""
    if not isinstance(args, dict):
        return "arguments must be an object"
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in args:
            return f"missing required argument '{key}'"
    if schema.get("additionalProperties") is False:
        for key in args:
            if key not in props:
                return f"unknown argument '{key}'"
    import re
    for key, value in args.items():
        spec = props[key]
        expected = _TYPES[spec["type"]]
        if not isinstance(value, expected):
            return f"argument '{key}' must be of type {spec['type']}"
        if spec["type"] == "string":
            if "maxLength" in spec and len(value) > spec["maxLength"]:
                return f"argument '{key}' is too long"
            if "minLength" in spec and len(value.strip()) < spec["minLength"]:
                return f"argument '{key}' is too short"
            if "enum" in spec and value not in spec["enum"]:
                return f"argument '{key}' must be one of {spec['enum']}"
            if "pattern" in spec and not re.match(spec["pattern"], value.strip().upper()):
                return f"argument '{key}' has an invalid format"
    return None


class CaseSupportServer:
    def __init__(self, memory: Optional[MemoryStore] = None):
        self.memory = memory
        self.audit: list[dict] = []

    # ---------------------------------------------------------------- JSON-RPC
    def handle(self, request: Any, caller: Caller) -> dict:
        rid = request.get("id") if isinstance(request, dict) else None
        if not isinstance(request, dict) or request.get("jsonrpc") != "2.0" or "method" not in request:
            return self._error(rid, INVALID_REQUEST, "Invalid JSON-RPC request")

        method, params = request["method"], request.get("params") or {}

        if method == "initialize":
            return self._ok(rid, {
                "protocolVersion": PROTOCOL_VERSION,
                "serverInfo": SERVER_INFO,
                "capabilities": {"tools": {"listChanged": False}},
            })
        if method == "tools/list":
            # Only advertise tools the caller is allowed to use.
            visible = [
                {k: t[k] for k in ("name", "description", "inputSchema")}
                for t in TOOL_DEFINITIONS if t["required_scope"] in caller.scopes
            ]
            return self._ok(rid, {"tools": visible})
        if method == "tools/call":
            return self._call_tool(rid, params, caller)
        return self._error(rid, METHOD_NOT_FOUND, f"Unknown method '{method}'")

    def handle_json(self, text: str, caller: Caller) -> str:
        try:
            req = json.loads(text)
        except json.JSONDecodeError:
            return json.dumps(self._error(None, PARSE_ERROR, "Parse error"))
        return json.dumps(self.handle(req, caller))

    # ------------------------------------------------------------------- tools
    def _call_tool(self, rid, params, caller: Caller) -> dict:
        name = params.get("name") if isinstance(params, dict) else None
        args = params.get("arguments", {}) if isinstance(params, dict) else {}
        tool = _TOOLS.get(name)
        if tool is None:
            self._log(name, caller, "unknown_tool")
            return self._error(rid, METHOD_NOT_FOUND, f"Unknown tool '{name}'")
        if tool["required_scope"] not in caller.scopes:
            self._log(name, caller, "forbidden")
            return self._error(rid, FORBIDDEN, f"Missing scope '{tool['required_scope']}'")
        problem = validate_arguments(tool["inputSchema"], args)
        if problem:
            self._log(name, caller, "invalid_params")
            return self._error(rid, INVALID_PARAMS, problem)

        handler: Callable = getattr(self, "_t_" + name.replace(".", "_"))
        try:
            content, is_error = handler(args, caller)
        except Exception as exc:                      # tool crashed: fail closed, no internals
            self._log(name, caller, "tool_exception")
            return self._error(rid, TOOL_FAILED, "The tool failed. No action was taken.")
        self._log(name, caller, "error" if is_error else "ok")
        return self._ok(rid, {
            "content": [{"type": "text", "text": json.dumps(content)}],
            "isError": is_error,
        })

    def _owns(self, caller: Caller, case_id: str) -> bool:
        return CASE_OWNERS.get(case_id.strip().upper()) == caller.student_id

    def _t_case_get_status(self, args, caller):
        case_id = args["case_id"].strip().upper()
        result = get_case_status(case_id)
        if not result.get("success") or not self._owns(caller, case_id):
            return {"error": f"No case found for {case_id}."}, True
        result = {k: v for k, v in result.items() if k != "student_name"}   # minimise output
        return result, False

    def _t_ticket_create(self, args, caller):
        case_id = args["case_id"].strip().upper()
        if not args["confirmed_by_student"]:
            return {"error": "Student confirmation is required before a ticket is created."}, True
        lookup = get_case_status(case_id)
        if not lookup.get("success") or not self._owns(caller, case_id):
            return {"error": f"No case found for {case_id}."}, True
        status = str(lookup["status"]).replace("_", " ").lower()
        if status not in {"in progress", "under review"}:
            return {"error": "A ticket can be created only for a case that is in progress or under review."}, True
        result = create_support_ticket(case_id, args["category"], args["description"])
        if not result.get("success"):
            return {"error": "; ".join(result.get("errors", ["ticket creation failed"]))}, True
        if self.memory is not None and self.memory.has_consent(caller.student_id):
            self.memory.remember_ticket(caller.student_id, result["ticket"], lookup["status"])
        return result["ticket"], False

    def _t_memory_get_history(self, args, caller):
        if self.memory is None:
            return {"error": "Memory is not enabled."}, True
        return self.memory.export_all(caller.student_id), False

    def _t_memory_forget(self, args, caller):
        if self.memory is None:
            return {"error": "Memory is not enabled."}, True
        if not args["confirm"]:
            return {"error": "Deletion not confirmed."}, True
        return {"deleted_records": self.memory.forget(caller.student_id)}, False

    # ----------------------------------------------------------------- helpers
    def _log(self, tool, caller, outcome):
        self.audit.append({"tool": tool, "student_ref": student_ref(caller.student_id), "outcome": outcome})

    @staticmethod
    def _ok(rid, result):
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    @staticmethod
    def _error(rid, code, message):
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}
