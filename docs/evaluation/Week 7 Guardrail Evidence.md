# Week 7 Guardrail Evidence

**Owner:** Judith  
**Implementation:** `src/guardrails.py`, merged into `src/bounded_agent.py`  
**Tests:** `tests/test_guardrails.py`, `tests/test_guardrail_trace.py`  
**Scenario set:** `docs/evaluation/scenarios.json` (`baseline-1`, 43 scenarios)

| Guardrail | Implementation | Test evidence | Representative scenario(s) |
|---|---|---|---|
| Request validation | `screen_input` rejects missing student identity, empty/non-text input, oversized text and prompt extraction; removes control characters and flags injection phrasing. It runs before model calls. | `test_no_signed_in_student_is_refused_only_when_required`, `test_empty_whitespace_and_non_text`, `test_length_limit_boundary`, `test_prompt_extraction_is_refused`, `test_normal_message_passes_and_is_cleaned` | UNA-04, EDG-04, EDG-05, ADV-03 |
| Tool-result validation and injection containment | `sanitize_tool_result` removes student names, validates structured values, bounds free text and withholds instruction-like tool content. Withheld text is checked against output echo. | `test_name_dropped_and_input_not_modified`, `test_instruction_like_text_is_withheld_and_findings_hold_no_content`, `test_malformed_structured_field_is_replaced`, `test_withheld_text_is_not_repeated` | ADV-05 |
| Output validation and groundedness | `screen_output` compares claims with tool-derived facts and flags unsupported actions, status claims, ticket IDs, promises, dates and prompt leaks. `safe_response` returns a bounded grounded alternative. | `test_action_claims`, `test_status_mismatch_and_closed_claim`, `test_ticket_claim_without_a_ticket`, `test_invented_references_and_canaries`, `test_promises_money_and_deadlines`, `test_safe_responses_are_grounded_and_never_repeat_the_blocked_text` | ADV-02, ADV-06, INC-02, OUT-01, OUT-02, OUT-03 |
| Mandatory student identity | `REQUIRE_AUTHENTICATED_STUDENT` is enabled in `src/bounded_agent.py`; requests without an identity are refused before provider invocation. | `test_unauthenticated_request_is_recorded`; `test_no_signed_in_student_is_refused_only_when_required` | UNA-04 |
| Tool crash handling | The tool-execution boundary records the exception type, transitions the run to a failed state and returns `TOOL_FAILURE_REPLY` instead of propagating an unsafe or empty result. | `test_tool_crash_returns_a_safe_reply_and_keeps_the_crash_evidence` | TLF-02 |
| Privacy-safe guardrail tracing | `src/observability.py` keeps only guardrail layer/codes/flags and finding field/reason in the persisted safe summary; it omits blocked text and student content. | `test_output_guardrail_is_recorded_with_layer_and_codes_but_no_text`, `test_tool_result_finding_records_reason_only`, `test_unauthenticated_request_is_recorded` | ADV-05, UNA-04 |

## Re-test evidence and limits

The deterministic post-guardrails run passed **43/43** scenarios with **0** safety-violation runs; baseline passed **28/43** with **7** safety-violation runs. See `docs/evidence/week 7/after_scripted_scripted.json` and `.csv`, compared with `baseline_scripted.json` and `.csv`. The run uses scripted providers: it verifies application guardrails against controlled replies, not the probability of live-model errors.

The separate three-repeat live evaluation passed **104/111** runs with **0** safety violations, versus the live baseline's **90/111** and 3 safety violations. Seven live runs missed non-safety criteria; provider failures occurred in six scenario IDs outside planned outages. See `docs/evidence/week 7/after_live_live.json`, `.csv` and the per-run trace directory. These live results are a single nondeterministic sample and should be repeated when provider availability is stable.
