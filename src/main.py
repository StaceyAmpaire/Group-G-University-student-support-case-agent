import os
import json

from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq

from tools import (
    TOOL_SCHEMAS,
    TOOL_FUNCTIONS,
)


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-3.7-flash"

gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# ============================================================
# SYSTEM INSTRUCTION
# ============================================================

SYSTEM_INSTRUCTION = """
You are a University Student Support Assistant.

TASK:
Help students with approved university support information
and information about their support cases.

ALLOWED INFORMATION:
Use only information provided by the application,
approved university sources, tools, and information explicitly
provided in the conversation.

Do not use general model knowledge to invent university-specific
information.

RULES:

1. Never invent case statuses, dates, policies, procedures,
   departments, or other university-specific information.

2. Never guess missing information.

3. If a required tool argument is missing, ask the student for it.
   Do not invent the value.

4. Use get_case_status when the student asks about an existing
   support case and provides a case ID.

5. Use create_support_ticket when a student has an existing case
   and requests a new simulated support ticket.

6. Do not claim that a case exists unless get_case_status confirms it.

7. Do not claim that a ticket was created unless
   create_support_ticket returns success=True.

8. If a tool reports an error, explain the result accurately.
   Do not hide or invent a successful outcome.

9. Do not make decisions about:
   - admissions
   - grades
   - fees
   - disciplinary matters
   - other high-impact university decisions

10. Do not escalate, close, resolve, modify, or delete cases.

11. Do not invent or use tools that are not provided by the
    application.

12. Stay within the university student-support scope.

13. The support ticket tool creates a simulated ticket only.
    It does not create a real university record.

RESPONSE STYLE:
Respond clearly, briefly, concisely, and politely.
"""


# ============================================================
# GEMINI TOOL DEFINITIONS
# ============================================================

gemini_function_declarations = [
    types.FunctionDeclaration(
        name=schema["name"],
        description=schema["description"],
        parameters_json_schema=schema["parameters"],
    )
    for schema in TOOL_SCHEMAS
]

gemini_tools = [
    types.Tool(
        function_declarations=gemini_function_declarations
    )
]


# ============================================================
# GROQ TOOL DEFINITIONS
# ============================================================

groq_tools = [
    {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema["description"],
            "parameters": schema["parameters"],
        },
    }
    for schema in TOOL_SCHEMAS
]


# ============================================================
# AUTHORIZED TOOL EXECUTION
# ============================================================

def execute_tool(function_name, arguments):
    """
    Execute only tools explicitly authorized by the application.

    The model does not directly execute Python functions.
    It can only request a tool by name.

    This allow-list is the application-level authorization boundary.
    """

    tool_function = TOOL_FUNCTIONS.get(function_name)

    if tool_function is None:
        return {
            "success": False,
            "error": (
                f"Tool '{function_name}' is not authorized "
                "by the application."
            ),
        }

    try:
        return tool_function(**arguments)

    except TypeError as error:
        return {
            "success": False,
            "error": f"Invalid tool arguments: {error}",
        }

    except Exception as error:
        return {
            "success": False,
            "error": (
                f"Unexpected tool execution error: {error}"
            ),
        }


# ============================================================
# GEMINI AGENT
# ============================================================

def run_with_gemini(student_message):
    """
    Run the agent using Gemini.

    Gemini is used as the fallback model when Groq fails.
    """

    print("\n[MODEL] Gemini fallback")
    print(f"[MODEL ID] {GEMINI_MODEL}")

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=student_message
                )
            ],
        )
    ]

    max_tool_rounds = 3

    for _ in range(max_tool_rounds):

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=gemini_tools,
            ),
        )

        candidate = response.candidates[0]
        parts = candidate.content.parts

        function_call_parts = [
            part
            for part in parts
            if part.function_call is not None
        ]

        # No tool requested
        if not function_call_parts:
            return response.text

        # Preserve Gemini's function-call message
        contents.append(candidate.content)

        tool_response_parts = []

        for part in function_call_parts:

            function_call = part.function_call

            function_name = function_call.name

            arguments = (
                dict(function_call.args)
                if function_call.args
                else {}
            )

            print("\n[TOOL SELECTED]")
            print(function_name)

            print("\n[TOOL ARGUMENTS]")
            print(json.dumps(arguments, indent=2))

            tool_result = execute_tool(
                function_name,
                arguments
            )

            print("\n[TOOL RESULT]")
            print(json.dumps(tool_result, indent=2))

            tool_response_parts.append(
                types.Part.from_function_response(
                    name=function_name,
                    response=tool_result,
                )
            )

        contents.append(
            types.Content(
                role="user",
                parts=tool_response_parts,
            )
        )

    return (
        "I was unable to complete the requested operation "
        "within the allowed tool-call limit."
    )


# ============================================================
# GROQ AGENT
# ============================================================

def run_with_groq(student_message):
    """
    Run the agent using Groq.

    Groq is the primary model.
    """

    print("\n[MODEL] Groq primary")
    print(f"[MODEL ID] {GROQ_MODEL}")

    messages = [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTION,
        },
        {
            "role": "user",
            "content": student_message,
        },
    ]

    max_tool_rounds = 3

    for _ in range(max_tool_rounds):

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=groq_tools,
            tool_choice="auto",
        )

        response_message = response.choices[0].message

        # Model answered without using a tool
        if not response_message.tool_calls:
            return response_message.content

        # Preserve the assistant's tool-call message
        messages.append(
            response_message.model_dump(
                exclude_none=True
            )
        )

        for tool_call in response_message.tool_calls:

            function_name = tool_call.function.name

            try:
                arguments = json.loads(
                    tool_call.function.arguments
                )
            except json.JSONDecodeError as error:

                tool_result = {
                    "success": False,
                    "error": (
                        "The model produced invalid tool "
                        f"arguments: {error}"
                    ),
                }

                arguments = {}

            else:

                print("\n[TOOL SELECTED]")
                print(function_name)

                print("\n[TOOL ARGUMENTS]")
                print(
                    json.dumps(
                        arguments,
                        indent=2
                    )
                )

                tool_result = execute_tool(
                    function_name,
                    arguments
                )

            print("\n[TOOL RESULT]")
            print(
                json.dumps(
                    tool_result,
                    indent=2
                )
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(
                        tool_result
                    ),
                }
            )

    return (
        "I was unable to complete the requested operation "
        "within the allowed tool-call limit."
    )


# ============================================================
# AGENT WITH MODEL FALLBACK
# ============================================================

def run_agent(student_message):
    """
    Primary model:
        Groq

    Fallback model:
        Gemini

    If Groq fails because of an unavailable service,
    API problem, rate limit, or another exception,
    the application attempts Gemini.
    """

    # --------------------------------------------------------
    # PRIMARY: GROQ
    # --------------------------------------------------------

    try:
        return run_with_groq(student_message)

    except Exception as groq_error:

        print("\n[GROQ FAILED]")
        print(groq_error)

        print("\n[FALLBACK]")
        print("Switching to Gemini...")

    # --------------------------------------------------------
    # FALLBACK: GEMINI
    # --------------------------------------------------------

    try:
        return run_with_gemini(student_message)

    except Exception as gemini_error:

        print("\n[GEMINI FAILED]")
        print(gemini_error)

        return (
            "I am currently unable to process your request. "
            "Please try again later."
        )


# ============================================================
# WEEK 4 DEMONSTRATION 1
# TOOL 1: GET CASE STATUS
# ============================================================

print("=" * 70)
print("DEMONSTRATION 1: GET CASE STATUS")
print("=" * 70)

question_1 = """
Can you check the status of CASE-1045?
"""

answer_1 = run_agent(question_1)

print("\nFINAL RESPONSE:")
print(answer_1)


# ============================================================
# WEEK 4 DEMONSTRATION 2
# TOOL 2: CREATE SUPPORT TICKET
# ============================================================

print("\n")
print("=" * 70)
print("DEMONSTRATION 2: CREATE SUPPORT TICKET")
print("=" * 70)

question_2 = """
I am having trouble completing registration for CASE-1045.
Please create a registration support ticket for me.
"""

answer_2 = run_agent(question_2)

print("\nFINAL RESPONSE:")
print(answer_2)


# ============================================================
# FAILURE / AUTHORIZATION TEST
# ============================================================

print("\n")
print("=" * 70)
print("FAILURE / AUTHORIZATION TEST")
print("=" * 70)

unauthorized_result = execute_tool(
    "delete_student_record",
    {
        "case_id": "CASE-1045"
    }
)

print("\nUNAUTHORIZED TOOL RESULT:")
print(
    json.dumps(
        unauthorized_result,
        indent=2
    )
)


# ============================================================
# UNKNOWN CASE TEST
# ============================================================

print("\n")
print("=" * 70)
print("FAILURE TEST: UNKNOWN CASE")
print("=" * 70)

unknown_case_result = execute_tool(
    "get_case_status",
    {
        "case_id": "CASE-9999"
    }
)

print("\nUNKNOWN CASE RESULT:")
print(
    json.dumps(
        unknown_case_result,
        indent=2
    )
)


# ============================================================
# MISSING PARAMETER TEST
# ============================================================

print("\n")
print("=" * 70)
print("FAILURE TEST: MISSING PARAMETER")
print("=" * 70)

missing_parameter_result = execute_tool(
    "create_support_ticket",
    {
        "case_id": "CASE-1045",
        "category": "registration",
    }
)

print("\nMISSING PARAMETER RESULT:")
print(
    json.dumps(
        missing_parameter_result,
        indent=2
    )
)