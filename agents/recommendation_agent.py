# from config import TOP_N

# def recommend_meals(state):
#     sorted_meals = sorted(state["scored_meals"], key=lambda x: x["score"], reverse=True)
#     state["final_recommendations"] = sorted_meals[:TOP_N]
#     return state

"""
Agent 4: Recommendation & Report Agent
=======================================

Role in the MAS Pipeline
--------------------------
This is the final agent in the Food Recommendation Multi-Agent System.
It receives the scored meals produced by Agent 3 (NutritionAnalyzerAgent),
selects the best ones, generates a structured Markdown report, and
persists both the report and structured JSON results to disk.

Persona & Design Philosophy
-----------------------------
This agent acts as a "nutritional journalist" — it takes raw scoring data
and turns it into something a real human can read, understand, and act on.
It reasons about diversity (don't recommend 5 Italian dishes), presentation
(rank badges, score bars), and explainability (why was each meal chosen).

Assignment Compliance
----------------------
✔  One distinct agent with a clear, non-overlapping role
✔  Uses two custom Python tools: report_tool.py and scoring_tool.py
✔  Reads from and writes to global state (FoodState TypedDict)
✔  Logs every step via the shared logger tool
✔  Works entirely locally — no API keys required
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List

from tools.logger import log_agent_step
from tools.report_tool import (
    build_markdown_report,
    save_markdown_report,
    save_json_results,
    print_console_summary,
)
from tools.scoring_tool import rank_meals, diversify_meals, enrich_with_rank


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
try:
    from config import TOP_N
except ImportError:
    TOP_N = 5


# ═════════════════════════════════════════════════════════════
# Public Entry Point — called by main.py / LangGraph node
# ═════════════════════════════════════════════════════════════

def recommend_meals(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agent 4 main function: Recommendation & Report Agent.

    Reads:
        state["scored_meals"]   — list of meal dicts scored by Agent 3
        state["preferences"]    — user preferences dict from Agent 1

    Writes:
        state["final_recommendations"]  — top-N enriched meal dicts
        state["logs"]                   — appends one log entry

    Side effects:
        - Saves a Markdown report to outputs/reports/
        - Saves a JSON results file to outputs/results/

    Args:
        state: The shared FoodState TypedDict passed through the pipeline.

    Returns:
        Updated state with final_recommendations populated.
    """
    print("\n[Agent 4] Starting Recommendation & Report Agent...")

    scored_meals: List[Dict[str, Any]] = state.get("scored_meals", [])
    preferences: Dict[str, Any] = state.get("preferences", {})

    # ── Step 1: Rank all scored meals by final_score DESC ──────
    ranked: List[Dict[str, Any]] = rank_meals(scored_meals)
    print(f"[Agent 4] Ranked {len(ranked)} scored meals.")

    # ── Step 2: Apply cuisine diversity then slice to TOP_N ────
    top_meals: List[Dict[str, Any]] = diversify_meals(ranked, top_n=TOP_N)
    print(f"[Agent 4] Selected top {len(top_meals)} diverse meals.")

    # ── Step 3: Enrich meals with rank numbers and badges ──────
    enriched: List[Dict[str, Any]] = enrich_with_rank(top_meals)

    # ── Step 4: Write to shared state ─────────────────────────
    state["final_recommendations"] = enriched

    # ── Step 5: Build Markdown report ─────────────────────────
    report_md: str = build_markdown_report(enriched, preferences)

    # ── Step 6: Save report and JSON to disk ──────────────────
    report_path: str = save_markdown_report(report_md)
    json_path: str = save_json_results(enriched, preferences)

    # ── Step 7: Log the agent step ────────────────────────────
    log_agent_step(
        state=state,
        agent_name="RecommendationReportAgent",
        input_data={
            "total_scored_meals": len(scored_meals),
            "top_n_requested": TOP_N,
            "preferences_summary": {
                "diet": preferences.get("diet"),
                "calorie_limit": preferences.get("calorie_limit"),
                "exclude": preferences.get("exclude", []),
                "cuisine": preferences.get("cuisine"),
            },
        },
        output_data={
            "recommendations_count": len(enriched),
            "top_meal_name": enriched[0].get("name", "Unknown") if enriched else None,
            "top_meal_score": enriched[0].get("final_score") if enriched else None,
            "report_path": report_path,
            "json_path": json_path,
            "status": "success" if enriched else "no_results",
        },
    )

    # ── Step 8: Print console summary ─────────────────────────
    print_console_summary(enriched, preferences)

    print(f"[Agent 4] Done. {len(enriched)} recommendations generated.")
    return state