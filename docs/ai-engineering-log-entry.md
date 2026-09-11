## Week 2 — Foundation Model Integration (Kawuma Fred)

**Date:** 2026-09-10
**Task:** Integrate Foundation Model into Application

**Model / provider chosen:** `openai/gpt-oss-120b` via Groq API (free tier, no billing attached). Selected because it's the strongest tool-calling-capable model available on our Groq account (131K context, `tools`/`structured_outputs`/`reasoning` support) 

**Integration:** Wired `model_client.py` (Groq client + `call_model()`) into `agent.py`'s bounded loop (Sense/Context → Plan/Decide → Act/Tool → Observe → Stop/Re-plan), using the 5 tool schemas defined in `tools.py`.

**Issues found and fixed during integration testing:**
1. `model_client.py` — model name had a typo (`openai?gpt-oss-120b` instead of `openai/gpt-oss-120b`), causing `model_not_found` errors.
2. `agent.py` — the assistant's proposed tool calls (`reply.tool_calls`, SDK objects) were being appended directly into the message history instead of being converted to plain dicts, which would have caused a JSON serialization error the first time the model actually invoked a tool.

**Verification:** Ran two manual test prompts (`src/agent.py` `__main__` block):
- No case ID given → model correctly asked for missing info instead of guessing (per AI Boundary Matrix / safety requirements), no tool call.
- Full student ID + case ID given → model called `get_case_status`, received the stub result, and produced a grounded final response. Confirms the full tool-calling loop works.


