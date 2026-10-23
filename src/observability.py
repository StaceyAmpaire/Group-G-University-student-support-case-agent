"""Privacy-conscious execution trace metadata for agent runs."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def begin_run(system_instruction: str, tool_schemas: list[dict], providers: list[Any]) -> dict:
    prompt_payload = json.dumps(
        {"system_instruction": system_instruction, "tools": tool_schemas},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    provider_config = [
        {
            "provider": getattr(provider, "name", type(provider).__name__),
            "model": getattr(provider, "model", None),
        }
        for provider in providers
    ]
    return {
        "run_id": uuid.uuid4().hex,
        "started_at": _utc_now(),
        "started_monotonic": time.perf_counter(),
        "prompt_version": "sha256:" + hashlib.sha256(prompt_payload).hexdigest()[:16],
        "providers": provider_config,
    }


def trace_event(trace: list[dict], event: str, **fields: Any) -> dict:
    entry = {"timestamp": _utc_now(), "event": event, **fields}
    trace.append(entry)
    return entry


def _safe_event(event: dict) -> dict:
    safe = {
        key: event[key]
        for key in ("timestamp", "event", "iteration", "provider", "model", "latency_ms")
        if key in event
    }
    if event.get("event") == "tool_call":
        result = event.get("result")
        safe.update({
            "tool": event.get("tool"),
            "input_fields": sorted((event.get("arguments") or {}).keys()),
            "dropped_argument_names": event.get("dropped_arguments", []),
            "success": bool(result.get("success")) if isinstance(result, dict) else False,
            "error_type": event.get("error_type"),
            "data_lookup": (
                {
                    "kind": "case_status",
                    "matched": bool(result.get("success")) if isinstance(result, dict) else False,
                }
                if event.get("tool") == "get_case_status"
                else None
            ),
        })
    elif event.get("event") == "provider_failure":
        safe["error_type"] = event.get("error_type")
    elif event.get("event") == "stop":
        safe["reason"] = event.get("reason")
        if "response" in event:
            safe["response_characters"] = len(event["response"])
    elif event.get("event") == "model_call":
        safe["tool_calls_requested"] = event.get("tool_calls_requested", 0)
    elif event.get("event") == "guardrail":
        for key in ("layer", "code", "codes", "flags"):
            if key in event:
                safe[key] = event[key]
        if "findings" in event:
            safe["findings"] = [
                {"field": item.get("field"), "reason": item.get("reason)}
                for item in event["findings"]
            ]
    return safe


def finish_run(context: dict, trace: list[dict], state: dict) -> dict:
    safe_events = [_safe_event(event) for event in trace]
    lookup_count = sum(
        1 for event in safe_events
        if (event.get("data_lookup") or {}).get("kind") == "case_status"
    )
    transitions = [
        {"from": item["from"], "to": item["to"]}
        for item in state.get("transitions", [])
    ]
    return {
        "schema_version": 1,
        "run_id": context["run_id"],
        "started_at": context["started_at"],
        "duration_ms": round((time.perf_counter() - context["started_monotonic"]) * 1000),
        "prompt_version": context["prompt_version"],
        "providers": context["providers"],
        "retrieval": {
            "document_retrieval_used": False,
            "case_status_lookups": lookup_count,
        },
        "outcome": {
            "phase": state.get("phase"),
            "iteration": state.get("iteration"),
            "transitions": transitions,
        },
        "events": safe_events,
    }
