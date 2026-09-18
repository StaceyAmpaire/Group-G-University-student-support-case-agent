# Group-G-University-student-support-case-agent

## Setup

### 1. Clone the repository

```powershell
git clone https://github.com/StaceyAmpaire/Group-G-University-student-support-case-agent.git
cd Group-G-University-student-support-case-agent
```

### 2. Create and activate virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Get your API keys

**Gemini API key:**
[https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey?utm_source=chatgpt.com)

**Groq API key:**
[https://console.groq.com/keys](https://console.groq.com/keys?utm_source=chatgpt.com)

Create a `.env` file in the project root and add:

```env
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

### 5. Run the RAG demo

```powershell
python src/rag_demo.py
```
