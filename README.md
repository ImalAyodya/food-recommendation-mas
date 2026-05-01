# Food Recommendation Multi-Agent System (MAS)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://github.com/langchain-ai/langgraph)
[![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A locally-running, privacy-first food recommendation system built with a
**four-agent pipeline** orchestrated by **LangGraph StateGraph** and powered
by **Ollama** (local LLMs — no API keys required).

---

## Architecture Overview

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  LangGraph StateGraph (workflow.py)                         │
│                                                             │
│  [Agent 1] PreferenceAnalyzerAgent                         │
│       └─ tools: preference_validation_tool, logger          │
│            ↓                                                │
│  [Agent 2] MenuFetcherAgent                                 │
│       └─ tools: data_loader, filter_tool, logger            │
│            ↓                                                │
│  [Agent 3] NutritionAnalyzerAgent                          │
│       └─ tools: nutrition_tool, logger                      │
│            ↓                                                │
│  [Agent 4] RecommendationReportAgent                       │
│       └─ tools: scoring_tool, report_tool, logger           │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
 Final Recommendations + Markdown Report + JSON Results + Trace
```

See [`docs/architecture.md`](docs/architecture.md) for the full Mermaid diagram.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python      | ≥ 3.11  |
| Ollama      | latest  |
| LLM model   | llama3.2 (or llama3 / mistral) |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/food-recommendation-mas.git
cd food-recommendation-mas
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Place the raw CSV file at:

```
data/raw/food_recommendation_dataset_100k.csv
```

Then run the preprocessor:

```bash
python data/preprocess.py
```

This generates `data/processed/meals.csv` (the file used by the pipeline).

---

## Ollama Setup

### Install Ollama

Download from [https://ollama.com/download](https://ollama.com/download).

### Pull the model

```bash
ollama pull llama3.2
```

### Start the Ollama server

```bash
ollama serve
```

The server runs at `http://localhost:11434` by default.  
You can verify it is running with:

```bash
curl http://localhost:11434
```

### Change the model (optional)

Edit `config.py`:

```python
OLLAMA_MODEL = "llama3.2"   # or "llama3", "mistral", "phi3", etc.
```

---

## Running the System

### CLI mode

```bash
python main.py
```

You will be prompted to enter your food preference, e.g.:

```
Enter your food preference: vegan food under 500 calories without nuts
```

Output files are saved to:

```
outputs/reports/   ← Markdown recommendation report
outputs/results/   ← JSON structured results
outputs/logs/      ← Full pipeline trace (run_id.json)
```

### Web UI mode

```bash
python run_web.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

> **Note:** The pipeline works even when Ollama is offline — Agent 1 and
> Agent 2 fall back to deterministic rule-based logic automatically.

---

## Running Tests

### Run all tests (quiet output)

```bash
pytest -q
```

### Run with verbose output

```bash
pytest -v
```

### Run a specific test file

```bash
pytest tests/test_nutrition_agent.py -v
pytest tests/test_tools.py -v
pytest tests/test_observability.py -v
pytest tests/test_workflow.py -v
pytest tests/test_preference_agent.py -v
pytest tests/test_recommendation_agent.py -v
```

### Test coverage (optional)

```bash
pip install pytest-cov
pytest --cov=. --cov-report=term-missing -q
```

---

## Project Structure

```
food-recommendation-mas/
├── agents/
│   ├── preference_agent.py        # Agent 1: Extracts dietary preferences via LLM
│   ├── menu_agent.py              # Agent 2: Filters meals from dataset
│   ├── nutrition_agent.py         # Agent 3: Scores meals nutritionally
│   └── recommendation_agent.py   # Agent 4: Ranks, diversifies, reports
├── tools/
│   ├── data_loader.py             # Loads the processed CSV
│   ├── filter_tool.py             # Filters DataFrame by preferences
│   ├── nutrition_tool.py          # Core nutrition scoring logic
│   ├── preference_validation_tool.py  # Schema + injection validation
│   ├── scoring_tool.py            # Ranking and diversity selection
│   ├── report_tool.py             # Markdown + JSON report generation
│   └── logger.py                  # Observability: logs, tool_calls, trace
├── tests/
│   ├── test_agents.py             # Pipeline smoke / integration tests
│   ├── test_nutrition_agent.py    # Agent 3 unit + integration tests
│   ├── test_preference_agent.py   # Agent 1 tests + injection tests
│   ├── test_tools.py              # Tool unit tests + injection tests
│   ├── test_workflow.py           # LangGraph workflow tests
│   ├── test_observability.py      # Logger / trace tests
│   ├── test_menu_agent.py         # Agent 2 tests
│   └── test_recommendation_agent.py  # Agent 4 tests
├── data/
│   ├── preprocess.py              # CSV preprocessor (relative paths)
│   ├── raw/                       # Place raw dataset here
│   └── processed/                 # Generated meals.csv goes here
├── web/
│   ├── app.py                     # Flask API (uses LangGraph workflow)
│   └── static/                    # HTML/CSS/JS for Web UI
├── outputs/
│   ├── reports/                   # Saved Markdown reports
│   ├── results/                   # Saved JSON results
│   └── logs/                      # Saved pipeline traces
├── docs/
│   └── architecture.md            # Mermaid architecture diagram
├── workflow.py                    # LangGraph StateGraph definition
├── state.py                       # FoodState TypedDict
├── config.py                      # OLLAMA_BASE_URL, OLLAMA_MODEL, TOP_N
├── main.py                        # CLI entry point
├── run_web.py                     # Web UI entry point
├── requirements.txt               # Python dependencies
├── CONTRIBUTIONS.md               # Team member contributions
└── README.md                      # This file
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | LLM model name |
| `TOP_N` | `5` | Number of final recommendations |
| `DATA_PATH` | `data/processed/meals.csv` | Path to processed dataset |

---

## Key Design Decisions

1. **LangGraph StateGraph** — Explicit pipeline graph with `START → Agent1 → Agent2 → Agent3 → Agent4 → END`. Every node receives and returns the full `FoodState`.

2. **FoodState TypedDict** — Single shared state object with clearly owned fields per agent, preventing cross-agent mutation.

3. **Prompt-injection hardening** — Agent 1 checks all user input with `detect_injection()` before forwarding to the LLM.

4. **Offline resilience** — Agents 1 and 2 fall back to deterministic rule-based logic when Ollama is unavailable.

5. **Full observability** — Every tool call is recorded in `state["tool_calls"]`, and the complete trace is persisted to `outputs/logs/<run_id>.json`.

---

## License

MIT — see [LICENSE](LICENSE).