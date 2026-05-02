"""
state.py
========
Shared state TypedDict for the Food Recommendation Multi-Agent System.

FoodState is passed between every agent node in the LangGraph pipeline.
Each agent reads from and writes to specific keys — no agent mutates
keys owned by another agent.

Key ownership:
    user_input        → set by caller (main.py / web/app.py)
    run_id            → set by caller before pipeline starts
    preferences       → PreferenceAnalyzerAgent (Agent 1)
    candidate_meals   → MenuFetcherAgent (Agent 2)
    nutrition_evaluations → NutritionAnalyzerAgent (Agent 3)
    scored_meals      → NutritionAnalyzerAgent (Agent 3)
    final_recommendations → RecommendationReportAgent (Agent 4)
    report_path       → RecommendationReportAgent (Agent 4)
    json_path         → RecommendationReportAgent (Agent 4)
    tool_calls        → appended by every agent via logger
    errors            → appended by any agent that catches an exception
    logs              → appended by every agent via log_agent_step
"""

from typing import TypedDict, List, Dict, Any, Optional


class FoodState(TypedDict):
    # ── Input ─────────────────────────────────────────────────
    user_input: str                          # Raw text from the user
    run_id: str                              # UUID for this pipeline run

    # ── Agent 1: Preference Analyzer ──────────────────────────
    preferences: Dict[str, Any]             # Structured diet/calorie prefs

    # ── Agent 2: Menu Fetcher ──────────────────────────────────
    candidate_meals: List[Dict[str, Any]]   # Raw meals from dataset

    # ── Agent 3: Nutrition Analyzer ────────────────────────────
    nutrition_evaluations: List[Dict[str, Any]]  # Per-meal eval records
    scored_meals: List[Dict[str, Any]]           # Accepted (non-rejected) meals

    # ── Agent 4: Recommendation & Report ──────────────────────
    final_recommendations: List[Dict[str, Any]]  # Top-N enriched meals
    report_path: Optional[str]              # Path to saved Markdown report
    json_path: Optional[str]               # Path to saved JSON results
    trace_path: Optional[str]

    # ── Observability ──────────────────────────────────────────
    tool_calls: List[Dict[str, Any]]        # Every tool invocation record
    errors: List[str]                       # Non-fatal errors / warnings
    logs: List[Dict[str, Any]]             # Agent step logs (for UI/audit)