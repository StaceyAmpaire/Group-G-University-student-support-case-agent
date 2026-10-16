"""
state.py  (Week 6)

Explicit workflow/session state for the Student-Support Case Agent.

Why: in Week 5 the run state was an ad-hoc dict with three keys. Week 6 makes
the workflow a small, named state machine so that
  * every run is always in exactly one documented phase,
  * only documented transitions are possible (illegal ones raise),
  * every transition is recorded for the execution trace.

SessionState is deliberately *session-scoped*: it lives for one agent run and is
discarded afterwards. Anything that must outlive the session goes through
memory.py, with consent, retention and deletion rules.

For backwards compatibility with the Week 5 code and tests, SessionState
supports dict-style access (state["ticket_created"], state.get("x")).
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Phase(str, Enum):
    INTAKE = "intake"                              # message received, nothing verified
    CASE_VERIFIED = "case_verified"                # get_case_status succeeded
    CASE_NOT_FOUND = "case_not_found"              # lookup failed / not owner
    AWAITING_CONFIRMATION = "awaiting_confirmation"  # duplicate warning raised
    TICKET_CREATED = "ticket_created"              # simulated ticket created
    RESPONDED = "responded"                        # terminal: final answer given
    LIMIT_REACHED = "limit_reached"                # terminal: iteration cap hit
    FAILED = "failed"                              # terminal: unrecoverable error


TERMINAL_PHASES = {Phase.RESPONDED, Phase.LIMIT_REACHED, Phase.FAILED}

# Allowed transitions. Anything not listed here is rejected.
TRANSITIONS: dict[Phase, set[Phase]] = {
    Phase.INTAKE: {
        Phase.CASE_VERIFIED, Phase.CASE_NOT_FOUND,
        Phase.RESPONDED, Phase.LIMIT_REACHED, Phase.FAILED,
    },
    Phase.CASE_VERIFIED: {
        Phase.CASE_VERIFIED, Phase.CASE_NOT_FOUND, Phase.AWAITING_CONFIRMATION,
        Phase.TICKET_CREATED, Phase.RESPONDED, Phase.LIMIT_REACHED, Phase.FAILED,
    },
    Phase.CASE_NOT_FOUND: {
        Phase.CASE_VERIFIED, Phase.CASE_NOT_FOUND,
        Phase.RESPONDED, Phase.LIMIT_REACHED, Phase.FAILED,
    },
    Phase.AWAITING_CONFIRMATION: {
        Phase.CASE_VERIFIED, Phase.TICKET_CREATED,
        Phase.RESPONDED, Phase.LIMIT_REACHED, Phase.FAILED,
    },
    Phase.TICKET_CREATED: {
        Phase.RESPONDED, Phase.LIMIT_REACHED, Phase.FAILED,
    },
    Phase.RESPONDED: set(),
    Phase.LIMIT_REACHED: set(),
    Phase.FAILED: set(),
}


class IllegalTransition(Exception):
    """Raised when code tries to move the workflow along an undocumented edge."""


def student_ref(student_id: Optional[str]) -> Optional[str]:
    """Short one-way reference used in traces so raw student IDs are not logged."""
    if not student_id:
        return None
    return "stu_" + hashlib.sha256(student_id.encode("utf-8")).hexdigest()[:10]


@dataclass
class SessionState:
    # ---- identity (supplied by the application, never by the model) ----
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    student_id: Optional[str] = None
    memory_enabled: bool = False        # True only if the student has consented

    # ---- workflow ----
    phase: Phase = Phase.INTAKE
    iteration: int = 0

    # ---- Week 5 fields (kept for compatibility) ----
    checked_case_id: Optional[str] = None
    case_result: Optional[dict] = None
    ticket_created: bool = False

    # ---- Week 6 fields ----
    ticket_id: Optional[str] = None
    memory_events: list = field(default_factory=list)   # reads/writes of memory
    transitions: list = field(default_factory=list)     # (from, to, reason)

    # ------------------------------------------------------------------
    def transition(self, new_phase: Phase, reason: str = "") -> None:
        if new_phase not in TRANSITIONS[self.phase]:
            raise IllegalTransition(
                f"{self.phase.value} -> {new_phase.value} is not an allowed transition"
            )
        self.transitions.append(
            {"from": self.phase.value, "to": new_phase.value,
             "iteration": self.iteration, "reason": reason}
        )
        self.phase = new_phase

    @property
    def is_terminal(self) -> bool:
        return self.phase in TERMINAL_PHASES

    def log_memory_event(self, action: str, detail: str) -> None:
        self.memory_events.append(
            {"action": action, "detail": detail, "iteration": self.iteration}
        )

    def snapshot(self) -> dict:
        """JSON-safe view for traces. Raw student_id is replaced by a hash reference."""
        return {
            "session_id": self.session_id,
            "student_ref": student_ref(self.student_id),
            "memory_enabled": self.memory_enabled,
            "phase": self.phase.value,
            "iteration": self.iteration,
            "checked_case_id": self.checked_case_id,
            "ticket_created": self.ticket_created,
            "ticket_id": self.ticket_id,
            "memory_events": list(self.memory_events),
            "transitions": list(self.transitions),
        }

    # ---- dict-style compatibility with Week 5 code/tests ----
    def __getitem__(self, key: str) -> Any:
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def advance(state: Any, new_phase: Phase, reason: str = "") -> None:
    """Transition only if `state` is a SessionState (legacy dict states are ignored)."""
    if isinstance(state, SessionState):
        state.transition(new_phase, reason)
