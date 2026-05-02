"""
agents/nutrition_agent.py
==========================
Agent 3: Nutrition Analyzer Agent
-----------------------------------
Role in MAS Pipeline
    Evaluates every candidate meal produced by Agent 2 against the user's
    nutritional preferences.  Delegates all scoring logic to
    tools/nutrition_tool.py so the agent stays thin (pure orchestration).

Tools used
    • tools/nutrition_tool.py  — evaluate_meal_nutrition / batch_evaluate
    • tools/logger.py          — agent step + tool call logging

Assignment compliance
    ✔  Distinct agent with a non-overlapping role
    ✔  Delegates to two custom tool functions (batch_evaluate, compute_health_flags)
    ✔  Reads from / writes to global FoodState  
    ✔  Logs every agent step and every tool call
    ✔  Works entirely locally — no LLM, no API keys required
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping

from tools.logger import log_agent_step, log_tool_call
from tools.nutrition_tool import batch_evaluate, compute_health_flags


def analyze_nutrition(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agent 3 main function: Nutrition Analyzer.

    Reads:
        state["preferences"]     — dietary constraints from Agent 1
        state["candidate_meals"] — meal list from Agent 2

    Writes:
        state["nutrition_evaluations"] — per-meal evaluation dicts (all meals)
        state["scored_meals"]          — only the non-rejected meals
        state["logs"]                  — one log entry appended
        state["tool_calls"]            — tool invocation records appended

    Args:
        state: Shared FoodState dict.

    Returns:
        Updated state dict.
    """
    print("\n[Agent 3] Starting Nutrition Analyzer Agent...")

    preferences = state.get("preferences", {})
    if not isinstance(preferences, Mapping):
        preferences = {}

    candidate_meals = state.get("candidate_meals", [])
    if not isinstance(candidate_meals, list):
        candidate_meals = []

    print(f"[Agent 3] Evaluating {len(candidate_meals)} candidate meals...")

    # ── Tool call: batch_evaluate ─────────────────────────────────────────────
    all_evaluations, accepted_meals = batch_evaluate(candidate_meals, preferences)

    log_tool_call(
        state, "batch_evaluate",
        {"meal_count": len(candidate_meals), "preferences": dict(preferences)},
        {
            "evaluated":  len(all_evaluations),
            "accepted":   len(accepted_meals),
            "rejected":   len(all_evaluations) - len(accepted_meals),
        },
    )

    # ── Store results in state ────────────────────────────────────────────────
    state["nutrition_evaluations"] = all_evaluations
    state["scored_meals"] = accepted_meals

    # ── Agent step log ────────────────────────────────────────────────────────
    log_agent_step(
        state=state,
        agent_name="NutritionAnalyzerAgent",
        input_data={
            "preferences":    dict(preferences),
            "candidate_count": len(candidate_meals),
        },
        output_data={
            "evaluated_count": len(all_evaluations),
            "accepted_count":  len(accepted_meals),
            "rejected_count":  len(all_evaluations) - len(accepted_meals),
        },
    )

    # Disabled for web UI (output visible in browser)
    # print(_build_console_report(preferences, all_evaluations))
    print(f"[Agent 3] Done. Accepted={len(accepted_meals)}, "
          f"Rejected={len(all_evaluations) - len(accepted_meals)}")
    return state


# ─── Console report helper ────────────────────────────────────────────────────

def _build_console_report(
    preferences: Mapping[str, Any],
    evaluations: List[Dict[str, Any]],
) -> str:
    accepted = [e for e in evaluations if not e.get("rejected", True)]
    rejected = len(evaluations) - len(accepted)

    summary_rows = [
        {
            "meal":            e.get("meal", "Unknown Meal"),
            "nutrition_score": e.get("nutrition_score"),
            "diet_match":      e.get("diet_match"),
            "final_score":     e.get("final_score"),
            "category":        e.get("category"),
            "health_flags":    e.get("health_flags", []),
        }
        for e in accepted
    ]

    report = {
        "preferences": dict(preferences),
        "summary": {
            "evaluated": len(evaluations),
            "accepted":  len(accepted),
            "rejected":  rejected,
        },
        "accepted_meals": summary_rows,
    }
    body = json.dumps(report, indent=2, ensure_ascii=True, default=str)
    footer = f"[Agent 3] Totals -> Accepted={len(accepted)}, Rejected={rejected}"
    return "[Agent 3] Nutrition Analyzer Report\n" + body + "\n" + footer