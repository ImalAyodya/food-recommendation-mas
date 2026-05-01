# Architecture — Food Recommendation Multi-Agent System

## System Overview

This system uses a **LangGraph StateGraph** to orchestrate four specialist AI agents
through a sequential pipeline. A single shared `FoodState` TypedDict is passed between
every node; each agent reads its inputs from state and writes its outputs back.

---

## Full Pipeline Diagram

```mermaid
flowchart TD
    USER([👤 User Input]) --> A1

    subgraph LANGGRAPH["LangGraph StateGraph (workflow.py)"]
        A1["🧠 Agent 1\nPreferenceAnalyzerAgent\nagents/preference_agent.py"]
        A2["🍽️ Agent 2\nMenuFetcherAgent\nagents/menu_agent.py"]
        A3["📊 Agent 3\nNutritionAnalyzerAgent\nagents/nutrition_agent.py"]
        A4["⭐ Agent 4\nRecommendationReportAgent\nagents/recommendation_agent.py"]

        A1 -->|preferences| A2
        A2 -->|candidate_meals| A3
        A3 -->|scored_meals + nutrition_evaluations| A4
    end

    subgraph TOOLS1["Agent 1 Tools"]
        T1A["preference_validation_tool.py\n• validate_preferences()\n• detect_injection()"]
        T1B["logger.py\n• log_agent_step()\n• log_tool_call()"]
    end

    subgraph TOOLS2["Agent 2 Tools"]
        T2A["data_loader.py\n• load_data()"]
        T2B["filter_tool.py\n• filter_meals()"]
        T2C["logger.py"]
    end

    subgraph TOOLS3["Agent 3 Tools"]
        T3A["nutrition_tool.py\n• evaluate_meal_nutrition()\n• batch_evaluate()\n• compute_health_flags()"]
        T3B["logger.py"]
    end

    subgraph TOOLS4["Agent 4 Tools"]
        T4A["scoring_tool.py\n• rank_meals()\n• diversify_meals()\n• enrich_with_rank()"]
        T4B["report_tool.py\n• build_markdown_report()\n• save_markdown_report()\n• save_json_results()"]
        T4C["logger.py\n• persist_trace()"]
    end

    subgraph OLLAMA["Ollama LLM (local)"]
        LLM["llama3.2\nhttp://localhost:11434"]
    end

    subgraph DATA["Dataset"]
        CSV["data/processed/meals.csv\n~100k rows"]
    end

    subgraph OUTPUTS["Output Artifacts"]
        RPT["outputs/reports/*.md\nMarkdown Report"]
        JSON["outputs/results/*.json\nJSON Results"]
        TRACE["outputs/logs/<run_id>.json\nFull Pipeline Trace"]
    end

    A1 -.-> T1A
    A1 -.-> T1B
    A1 -.->|LLM call| LLM

    A2 -.-> T2A
    A2 -.-> T2B
    A2 -.-> T2C
    A2 -.->|LLM call| LLM
    T2A -.->|reads| CSV

    A3 -.-> T3A
    A3 -.-> T3B

    A4 -.-> T4A
    A4 -.-> T4B
    A4 -.-> T4C
    T4B -.->|writes| RPT
    T4B -.->|writes| JSON
    T4C -.->|writes| TRACE

    A4 --> RESULT([📋 Final Recommendations])

    style LANGGRAPH fill:#1a1a2e,color:#eee,stroke:#7c3aed
    style TOOLS1 fill:#0f3460,color:#eee,stroke:#16213e
    style TOOLS2 fill:#0f3460,color:#eee,stroke:#16213e
    style TOOLS3 fill:#0f3460,color:#eee,stroke:#16213e
    style TOOLS4 fill:#0f3460,color:#eee,stroke:#16213e
    style OLLAMA fill:#16213e,color:#eee,stroke:#7c3aed
    style DATA fill:#16213e,color:#eee,stroke:#7c3aed
    style OUTPUTS fill:#16213e,color:#eee,stroke:#7c3aed
```

---

## State Flow (FoodState)

```mermaid
stateDiagram-v2
    [*] --> Initial: make_initial_state()

    Initial --> AfterAgent1: analyze_preferences()
    note right of AfterAgent1
        + user_input (str)
        + preferences (dict)
        + tool_calls updated
        + logs updated
    end note

    AfterAgent1 --> AfterAgent2: fetch_menu()
    note right of AfterAgent2
        + candidate_meals (list)
        + preferences (normalised)
    end note

    AfterAgent2 --> AfterAgent3: analyze_nutrition()
    note right of AfterAgent3
        + nutrition_evaluations (list)
        + scored_meals (list)
    end note

    AfterAgent3 --> AfterAgent4: recommend_meals()
    note right of AfterAgent4
        + final_recommendations (list)
        + report_path (str)
        + json_path (str)
        + trace persisted to disk
    end note

    AfterAgent4 --> [*]: Pipeline complete
```

---

## Agent Responsibilities

| Agent | Role | LLM Used? | Key Tool(s) |
|-------|------|-----------|-------------|
| **Agent 1** | Parse free-text user query → structured preferences | ✅ (with fallback) | `preference_validation_tool` |
| **Agent 2** | Normalise preferences + filter 100k-row dataset | ✅ (with fallback) | `filter_tool`, `data_loader` |
| **Agent 3** | Score each candidate meal nutritionally | ❌ rule-based | `nutrition_tool` |
| **Agent 4** | Rank, diversify, generate report + trace | ❌ rule-based | `scoring_tool`, `report_tool` |

---

## Web API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /` | GET | Serve Web UI (index.html) |
| `POST /api/recommend` | POST | Run full pipeline, return recommendations |
| `GET /api/status` | GET | Check Ollama + dataset availability |
| `GET /api/report/<run_id>` | GET | Download Markdown report for a run |

---

## Security Design

- **Prompt injection detection** in Agent 1 using 10 regex patterns before any LLM call
- **Schema validation** via `preference_validation_tool.py` — all fields type-checked and range-validated
- **Word-boundary matching** in exclusion and diet checks — prevents false positives (e.g., "nut" ≠ "peanut")
- **Local-only LLM** — no data leaves the machine; Ollama runs entirely on-device
