# Contributions

## Team Members & Responsibilities

| Member | Student ID | Agents / Components | Specific Contributions |
|--------|-----------|---------------------|----------------------|
| **Member 1** | IT21XXXXXX | Agent 1 (Preference Analyzer) | `agents/preference_agent.py`, `tools/preference_validation_tool.py`, `tests/test_preference_agent.py`, prompt-injection hardening, LLM integration & fallback logic |
| **Member 2** | IT21XXXXXX | Agent 2 (Menu Fetcher) | `agents/menu_agent.py`, `tools/filter_tool.py`, `tools/data_loader.py`, `data/preprocess.py`, dataset integration, LLM-based preference normalisation |
| **Member 3** | IT21XXXXXX | Agent 3 (Nutrition Analyzer) | `agents/nutrition_agent.py`, `tools/nutrition_tool.py`, `tests/test_nutrition_agent.py`, scoring algorithm design, diet-compliance logic |
| **Member 4** | IT21XXXXXX | Agent 4 (Recommendation & Report) | `agents/recommendation_agent.py`, `tools/scoring_tool.py`, `tools/report_tool.py`, `tests/test_recommendation_agent.py`, Markdown/JSON report generation |
| **All Members** | — | Orchestration & Infrastructure | `workflow.py` (LangGraph StateGraph), `state.py` (FoodState), `tools/logger.py` (observability), `web/app.py` (Flask UI), `tests/test_workflow.py`, `tests/test_observability.py`, `tests/test_tools.py`, `README.md` |

---

## Contribution Details

### Agent 1 — Preference Analyzer (Member 1)
- Designed the LLM prompt template for preference extraction
- Implemented deterministic fallback rules (no Ollama dependency)
- Built `preference_validation_tool.py` with JSON schema validation
- Implemented prompt-injection detection using regex patterns
- Wrote 30+ tests covering LLM path, fallback path, injection, and edge cases

### Agent 2 — Menu Fetcher (Member 2)
- Implemented LLM-based preference normalisation (structured JSON output)
- Built `filter_tool.py` with Pandas-based dataset filtering (6 filter dimensions)
- Fixed `data/preprocess.py` to use portable relative file paths
- Ensured fallback when LLM returns malformed output via `_merge_with_fallback`

### Agent 3 — Nutrition Analyzer (Member 3)
- Designed the multi-factor nutrition scoring algorithm (calorie, protein, carb, fat weights)
- Built `nutrition_tool.py` separating scoring logic from agent orchestration
- Implemented health flag system (6 warning categories)
- Wrote diet-compliance checks for vegan/vegetarian/keto with word-boundary matching
- Wrote 25+ unit and integration tests for the nutrition tool

### Agent 4 — Recommendation & Report (Member 4)
- Implemented cuisine diversity algorithm (max 2 per cuisine in top-N)
- Built `report_tool.py` generating detailed Markdown reports with score tables and statistics
- Built JSON results output with metadata for downstream consumption
- Integrated `persist_trace()` to flush the full pipeline trace to disk
- Wrote 30+ tests covering scoring, diversification, report generation, and edge cases

### Shared Infrastructure (All Members)
- Designed the `FoodState` TypedDict with 12 typed fields
- Built the LangGraph `StateGraph` pipeline (workflow.py)
- Implemented `log_tool_call()` and `persist_trace()` for full observability
- Built the Flask web UI with real-time pipeline step display
- Set up project structure, `.gitignore`, output directories, and CI test configuration

---

## Version Control

All team members contributed via Git with feature branches:
- `feature/agent1-preference`
- `feature/agent2-menu`
- `feature/agent3-nutrition`
- `feature/agent4-recommendation`
- `feature/langgraph-orchestration`
- `feature/observability`
- `feature/tests`

Pull requests were reviewed by at least one other team member before merging to `main`.
