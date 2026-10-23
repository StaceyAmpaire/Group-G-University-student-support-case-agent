# Week 7 Tracing and Privacy Note

**Owner:** Fred (observability); Stacey (evaluation integration)
**Implementation:** `src/observability.py`, integrated in `src/bounded_agent.py` and `src/run_evaluation.py`

Each run summary includes a unique run ID, UTC start time, total elapsed latency, a SHA-256 prompt/tool-schema version, provider and model configuration, retrieval counts, workflow outcome and a compact event list. Events record model/tool latency, tool outcome, provider failures, stop reason and guardrail codes. The evaluation runner persists both the aggregate result and a separate JSON summary for each scenario/repeat under `docs/evidence/<label>_<mode>_traces/`. When a run crashes before a safe summary is available, the per-run file explicitly records `trace_status: unavailable` and the exception type only.

The persisted observability summary is allow-list based. It omits student names, student IDs, message content, tool arguments and tool-result text. Tool-call events keep tool name, input field names, success/error type and a case-lookup outcome; guardrail events keep codes or finding field/reason, never blocked text. The aggregate evaluation result files are broader test evidence and include synthetic scenario messages and IDs; do not place real student records in the evaluation set.

## Sample traces

- Success: [sample_trace_success.json](<../evidence/week 7/sample_traces/sample_trace_success.json>) — NOR-01 completes a case-status lookup.
- Failure: [sample_trace_failure.json](<../evidence/week 7/sample_traces/sample_trace_failure.json>) — TLF-02 records a simulated tool exception and failed outcome without exposing the exception message.
- Fallback: [sample_trace_fallback.json](<../evidence/week 7/sample_traces/sample_trace_fallback.json>) — TLF-04 records the planned primary-provider outage and successful Gemini fallback.

These examples are from scripted runs. Provider and model values in scripted mode identify the simulated providers; their latency values are not live-provider performance measurements.
