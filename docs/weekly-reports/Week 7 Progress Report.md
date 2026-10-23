# Week 7 Progress Report

**Project:** University Student Support Case Agent  
**Group:** Group G — Makerere University  
**Week:** 7 — Evaluation, observability and guardrails  
**Repository branch:** `main`  
**Evidence baseline:** Scenario set `baseline-1`, 43 scenarios

## Work completed

- Desmond's reviewed scenario set and evaluation criteria documents are integrated on `main`: `docs/evaluation/scenarios.json`, `Dataset Review Signoff final.docx`, and `Evaluation Criteria final.docx`.
- Stacey's baseline evidence is available in `docs/evidence/week 7/`. The scripted baseline passed 28/43 scenarios, with 7 safety-violation runs. The live baseline was 90/111 across 37 live-eligible scenarios, with 3 safety-violation runs.
- Judith's input/output guardrails, tool-result sanitization, mandatory student identity check, and safe tool-crash handling are integrated with tests. The implementation maps to evidence in `docs/evaluation/Week 7 Guardrail Evidence.md`.
- Fred's observability work is integrated. The evaluator now saves one privacy-filtered trace per run. Run IDs, prompt versions, provider/model fields, latency, workflow outcome and guardrail events are captured without student names or message text in the trace summary. Details and sample traces are in `docs/evaluation/Week 7 Tracing and Privacy Note.md`.
- Stacey integrated both Week 7 branches onto `main`, resolved the observability merge, ran the full tests and completed after-guardrail evaluations. The scripted result is 43/43 with 0 safety violations. The live three-repeat result is 104/111 (93.7%) with 0 safety violations.
- Racheal's Failure Catalogue has been prepared as an evidence-backed draft with seven baseline failures, remediation references, scripted re-tests, live follow-up results and remaining issues: `docs/evaluation/Week 7 Failure Catalogue - Racheal draft.md`.

## Engineering decisions and remaining limitations

The scripted evaluation deliberately uses controlled model replies to test whether application-side controls stop unsafe and unsupported behaviour; it is not a live-model performance estimate. Live testing is nondeterministic: seven runs missed non-safety criteria, and provider failures affected results. Six scenario IDs recorded provider failures outside planned outage scenarios. Live latency ranged from 0 to 483,390 ms (median 2,743 ms; p95 12,976 ms), so the slowest result reflects provider/fallback instability, not normal response time. No live safety violations were recorded in the re-test.

Evaluation evidence is synthetic. Persisted observability summaries exclude student names and message text, while aggregate evaluation records retain synthetic scenario messages/IDs for reproducibility. The final live failures and provider-outage scenarios should be rerun once provider availability is stable. No ClickUp URL was present in the repository evidence; add the group's board/task links before external submission.

## Individual contribution and evidence

| Member | Owned work | Evidence |
|---|---|---|
| Desmond Cole | Dataset review/sign-off and six-criterion evaluation criteria | `docs/evaluation/scenarios.json`; `Dataset Review Signoff final.docx`; `Evaluation Criteria final.docx` |
| Stacey | Baseline runs, branch integration, evaluation/trace integration, scripted and live re-tests | Commits `4e53997`, `423f1b7`, `4968464`; `docs/evidence/week 7/` |
| Judith | Guardrail implementation, bounded-agent hooks and tests | Commit `048cdce`; `src/guardrails.py`; `tests/test_guardrails.py`; `tests/test_guardrail_trace.py` |
| Fred | Observability implementation and trace evidence | `src/observability.py`; `docs/evidence/week 7/sample_traces/`; `docs/evaluation/Week 7 Tracing and Privacy Note.md` |
| Racheal Mirembe | Failure Catalogue | Draft at `docs/evaluation/Week 7 Failure Catalogue - Racheal draft.md`; review evidence and complete hand-in checks |

## Verification and next steps

The complete offline test suite passes: **99 tests**. The after-scripted results pass all 43 scenarios; the live results include the residual failures and provider caveats documented above. Before final submission, Racheal should review and sign off the catalogue, the group should add ClickUp links, and the team should rerun the affected live cases if the providers are stable. Week 8 work should then package a reproducible release, final engineering report, evidence appendices and demonstration.
