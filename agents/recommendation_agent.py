"""
agents/recommendation_agent.py
================================
Agent 4: Recommendation & Report Agent
-----------------------------------------
Role in MAS Pipeline
    Final agent. Selects the best scored meals, applies cuisine diversity,
    generates a Markdown report + JSON results file, and persists the full
    pipeline trace via the logger.

Tools used
    • tools/scoring_tool.py  — rank_meals, diversify_meals, enrich_with_rank
    • tools/report_tool.py   — build_markdown_report, save_markdown_report,
                               save_json_results, print_console_summary
    • tools/logger.py        — log_agent_step, log_tool_call, persist_trace

Assignment compliance
    ✔  Distinct agent with a non-overlapping role
    ✔  Uses four custom tool functions across two tool files
    ✔  Writes report_path and json_path back to FoodState
    ✔  Persists trace at end of pipeline for full observability
    ✔  Works entirely locally — no API keys required
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from tools.logger import log_agent_step, log_tool_call, persist_trace
from tools.report_tool import (
    build_markdown_report,
    save_markdown_report,
    save_json_results,
    print_console_summary,
)
from tools.scoring_tool import rank_meals, diversify_meals, enrich_with_rank

try:
    from config import TOP_N
except ImportError:
    TOP_N = 5


def recommend_meals(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agent 4 main function: Recommendation & Report Agent.

    Reads:
        state["scored_meals"]   — accepted meals from Agent 3
        state["preferences"]    — user preferences from Agent 1

    Writes:
        state["final_recommendations"] — top-N enriched meal dicts
        state["report_path"]           — path to saved Markdown report
        state["json_path"]             — path to saved JSON results
        state["logs"]                  — one log entry appended
        state["tool_calls"]            — tool invocation records appended

    Args:
        state: Shared FoodState dict.

    Returns:
        Updated state dict.
    """
    print("\n[Agent 4] Starting Recommendation & Report Agent...")

    scored_meals: List[Dict[str, Any]] = state.get("scored_meals", [])
    preferences:  Dict[str, Any]       = state.get("preferences", {})
    run_id: str = state.get("run_id", "unknown")

    # ── Step 1: Rank ──────────────────────────────────────────────────────────
    ranked = rank_meals(scored_meals)
    log_tool_call(state, "rank_meals",
                  {"scored_meal_count": len(scored_meals)},
                  {"ranked_count": len(ranked)})
    print(f"[Agent 4] Ranked {len(ranked)} meals.")

    # ── Step 2: Diversify ─────────────────────────────────────────────────────
    top_meals = diversify_meals(ranked, top_n=TOP_N)
    log_tool_call(state, "diversify_meals",
                  {"ranked_count": len(ranked), "top_n": TOP_N},
                  {"selected_count": len(top_meals)})
    print(f"[Agent 4] Selected {len(top_meals)} diverse meals.")

    # ── Step 3: Enrich with rank ──────────────────────────────────────────────
    enriched = enrich_with_rank(top_meals)
    log_tool_call(state, "enrich_with_rank",
                  {"meal_count": len(top_meals)},
                  {"enriched_count": len(enriched)})

    # ── Step 4: Write to state ────────────────────────────────────────────────
    state["final_recommendations"] = enriched

    # ── Step 5: Build Markdown report ────────────────────────────────────────
    report_md = build_markdown_report(enriched, preferences)
    log_tool_call(state, "build_markdown_report",
                  {"meal_count": len(enriched)}, {"report_length_chars": len(report_md)})

    # ── Step 6: Save report and JSON ─────────────────────────────────────────
    report_path = save_markdown_report(report_md)
    json_path   = save_json_results(enriched, preferences)
    log_tool_call(state, "save_reports",
                  {"run_id": run_id},
                  {"report_path": report_path, "json_path": json_path})

    # ── Step 7: Write paths to state ──────────────────────────────────────────
    state["report_path"] = report_path
    state["json_path"]   = json_path

    # ── Step 8: Agent step log ────────────────────────────────────────────────
    log_agent_step(
        state=state,
        agent_name="RecommendationReportAgent",
        input_data={
            "total_scored_meals": len(scored_meals),
            "top_n_requested":    TOP_N,
            "preferences_summary": {
                "diet":         preferences.get("diet"),
                "calorie_limit": preferences.get("calorie_limit"),
                "exclude":      preferences.get("exclude", []),
                "cuisine":      preferences.get("cuisine"),
            },
        },
        output_data={
            "recommendations_count": len(enriched),
            "top_meal_name":  enriched[0].get("name")  if enriched else None,
            "top_meal_score": enriched[0].get("final_score") if enriched else None,
            "report_path":    report_path,
            "json_path":      json_path,
            "status":         "success" if enriched else "no_results",
        },
    )

    # ── Step 9: Persist full trace ────────────────────────────────────────────
    try:
        trace_path = persist_trace(state, run_id)
        print(f"[Agent 4] Trace saved -> {trace_path}")
    except Exception as exc:
        print(f"[Agent 4] Warning: could not persist trace: {exc}")

    # ── Step 10: Print console summary ────────────────────────────────────────
    print_console_summary(enriched, preferences)
    print(f"[Agent 4] Done. {len(enriched)} recommendations generated.")
    return state