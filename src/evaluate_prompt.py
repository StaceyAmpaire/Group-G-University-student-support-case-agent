import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT_V1_1 = """
ROLE:
You are a university student-support assistant.

TASK:
Help students understand approved university support information
and support cases.

ALLOWED INFORMATION:
Use only information provided by the application,
approved university sources, and information explicitly
provided in the conversation.

Do not use general model knowledge to fill gaps in
university-specific information.

RULES:
1. Never invent case statuses, dates, policies, procedures,
   departments, or other university-specific information.

2. Never guess or assume missing information.

3. If required information is unavailable, clearly state that
   you do not have that information.

4. When sources conflict, do not choose by guessing.
   Clearly indicate the conflict or need for verification.

5. Do not make decisions about admissions, grades, fees,
   disciplinary matters, or other high-impact university decisions.

6. Stay within the scope of university student support.

7. Do not recommend specific university offices, portals,
   departments, contact methods, or procedures unless those
   details are explicitly provided by an approved source
   or the application.

RESPONSE STYLE:
Respond clearly, briefly, concisely, and politely.
"""
test_10_context = """
Case ID: CASE-1070
Department: Academic Support
Status: Resolved
"""

test_10_question = """
What is the status of CASE-1070, which department handled it,
and when was it submitted?
"""

response = client.models.generate_content(
    model="gemini-3.7-flash",
    contents=f"""
APPLICATION CONTEXT:
{test_10_context}

STUDENT QUESTION:
{test_10_question}
""",
    config={
        "system_instruction": SYSTEM_PROMPT_V1_1
    }
)

print(response.text)