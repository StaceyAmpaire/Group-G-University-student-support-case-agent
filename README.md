# Group G — University Student Support Case Agent

A university student-support case agent developed as an academic software engineering project. The repository includes a command-line demo, a RAG demo, a separate memory demo, and evaluation scripts.

## Setup

### 1. Clone the repository

```powershell
git clone https://github.com/StaceyAmpaire/Group-G-University-student-support-case-agent.git
cd Group-G-University-student-support-case-agent
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure API keys

Create API keys for [Google Gemini](https://aistudio.google.com/app/apikey) and [Groq](https://console.groq.com/keys).

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

## Run the demos

**Main case-agent demo**

```powershell
python src/main.py
```

**Additional case-agent features**

```powershell
python src/main.py --extras
```

**Memory/MCP demo (offline)**

```powershell
python src/main.py --memory
```

**RAG demo**

```powershell
python src/rag_demo.py
```

The main demo and `--extras` require configured API keys. The memory/MCP demo is separate from the ordinary case-agent flow.

## Evaluation

Run the tests:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Run the offline scripted evaluation:

```powershell
python src/run_evaluation.py --mode scripted --label final_scripted
```

Run the live evaluation:

```powershell
python src/run_evaluation.py --mode live --repeats 3 --label final_live
```

Live evaluation requires API access and may incur charges. Evaluation outputs are written under `docs/evidence/`.

## Project limitations

- The case data and tickets are synthetic.
- The application is a local academic prototype, not a production service.
- RAG and the memory/MCP demo are separate from the ordinary case-agent flow.
- No live university service or ticketing system is connected.
- No hosted deployment is claimed.

## Submission documents

The final report and presentation are **not yet listed here as included files**. 
