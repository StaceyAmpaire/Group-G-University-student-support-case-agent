"""
Foundation model client for the University Student-Support Case Agent.

Wraps Groq's OpenAI-compatible API. The model is used ONLY for:
  - interpreting the student's natural-language request
  - deciding which approved tool to call (function calling)
  - drafting grounded explanations from retrieved evidence

It is never used to directly execute validation, authorization,
record access, or high-impact decisions. Those stay deterministic
(see the AI Boundary Matrix).
"""

import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq

MODEL_NAME = "openai/gpt-oss-120b"


def get_client() -> Groq:
    """Create a Groq client using GROQ_API_KEY from the environment."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Create a .env file (see .env.example) "
            "or export it in your shell before running the agent."
        )
    return Groq(api_key=api_key)


def call_model(client: Groq, messages: list[dict], tools: list[dict] | None = None,
                tool_choice: str = "auto", temperature: float = 0.2):
    """
    Send a chat completion request, optionally with tool/function definitions.

    Low temperature (0.2) is deliberate: this model is choosing between a
    small set of approved tools and drafting grounded text, not doing
    open-ended creative generation. We want it predictable.
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice if tools else None,
        temperature=temperature,
    )
    return response.choices[0].message
