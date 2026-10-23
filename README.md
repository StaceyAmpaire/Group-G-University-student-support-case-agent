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

## Run the bounded support-case agent

The main application demo uses `src/bounded_agent.py`, including the Week 7 guardrails:

```powershell
python src/main.py
python src/main.py --extras
python src/main.py --memory
```

The first two commands use the configured live model providers and require the API keys above. `--memory` runs the offline Week 6 memory/MCP demonstration. `src/agent_loop.py` preserves the Week 5 implementation for historical comparison and regression tests; it is not the current `main.py` runtime path.

## Week 7 evaluation

Run the deterministic, offline scenario set without API calls:

```powershell
python -m unittest discover -s tests -p "test_*.py"
python src/run_evaluation.py --mode scripted --label after_scripted
```

For a live provider evaluation, configure the API keys above and run:

```powershell
python src/run_evaluation.py --mode live --repeats 3 --label after_live
```

Live runs call the configured model providers and may incur API usage charges. Each evaluation writes summary JSON and CSV files plus one privacy-filtered trace JSON per run under `docs/evidence/<label>_<mode>_traces/`. The scenario results and traces contain synthetic scenario inputs; the observability summaries omit student names and message text.

## Week 8 submission and release rehearsal

The submission report and presentation are in `docs/Final Engineering Report.docx` and `docs/Final Presentation.pptx`. The report describes the project as a local academic prototype; it does not claim a hosted deployment or production readiness.

Before presenting or packaging a local release:

1. Use a clean clone and a virtual environment, then install `requirements.txt`.
2. Run `python -m unittest discover -s tests -p "test_*.py"` and review the checked-in evaluation evidence.
3. For an offline demonstration, run `python src/run_evaluation.py --mode scripted --label final_scripted`.
4. For the live case-agent demonstration, configure the required keys in an untracked `.env`, confirm provider access, and run `python src/main.py`.
5. Keep the RAG demo (`python src/rag_demo.py`) and Week 6 memory demo (`python src/main.py --memory`) clearly identified as separate demonstrations.

The final Week 7 evidence records 43/43 scripted passes after guardrails (28/43 at baseline) and 104/111 live passes over three repeats (90/111 at baseline). Safety-violation flags fell from seven to zero scripted and from three to zero live. Baseline evidence records commit `1687580`; after-guardrail evidence records `4968464`. Live runs remain variable: the after run had seven non-safety misses, unexpected provider failures, and a maximum latency of 483,390 ms. The scripted evaluator is not a live-model reliability measure.

Known submission boundaries: case data and tickets are synthetic; the CLI demo is not a production authentication system; RAG is not integrated into the main case-agent flow; persistent memory is exercised by the separate Week 6 demo rather than automatically enabled by the ordinary case-agent entry point; and no university service or real ticket system is connected. Hosting remains undecided. Do not publish a deployment claim or version tag until the team has reviewed the target commit, artifacts, privacy controls and operational requirements.
