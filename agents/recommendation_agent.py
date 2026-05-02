"""
agents/recommendation_agent.py

Agent 4: RecommendationReportAgent

Persona:
    This agent acts as the final recommendation packaging and reporting agent.
    It does not invent meals, change nutrition values, or calculate new nutrition
    scores. It only uses already scored meals from Agent 3 and converts them into
    ranked, diversified, explainable, and saved outputs.

Constraints:
    - Must only use state["scored_meals"] from Agent 3.
    - Must not create new meals.
    - Must not modify calorie, protein, fat, carb, or final_score values.
    - Must rank meals using final_score.
    - Must save Markdown and JSON outputs locally.
    - Must log agent inputs, tool calls, outputs, and errors.
"""

from __future__ import annotations

from typing import Any, Dict, List

from tools.logger import log_agent_step, log_tool_call, persist_trace
from tools.description_tool import generate_selection_description
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


AGENT_NAME = "RecommendationReportAgent"


def _ensure_agent4_state(state: Dict[str, Any]) -> None:
    """
    Validate and initialise the minimum state keys required by Agent 4.

    Args:
        state: Shared FoodState dictionary.

    Raises:
        TypeError: If state is not a dictionary or scored_meals/preferences
                   have invalid types.
    """
    if not isinstance(state, dict):
        raise TypeError("Agent 4 expects state to be a dictionary.")

    state.setdefault("preferences", {})
    state.setdefault("scored_meals", [])
    state.setdefault("final_recommendations", [])
    state.setdefault("tool_calls", [])
    state.setdefault("logs", [])
    state.setdefault("errors", [])

    if not isinstance(state["preferences"], dict):
        raise TypeError("state['preferences'] must be a dictionary.")

    if not isinstance(state["scored_meals"], list):
        raise TypeError("state['scored_meals'] must be a list.")


def recommend_meals(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agent 4 main function: rank, diversify, report, save, and log recommendations.

    Reads:
        state["scored_meals"] - scored meal list from NutritionAnalyzerAgent.
        state["preferences"]  - structured preferences from PreferenceAnalyzerAgent.

    Writes:
        state["final_recommendations"] - ranked and enriched top meals.
        state["report_path"]           - saved Markdown report path.
        state["json_path"]             - saved JSON result path.
        state["trace_path"]            - saved trace/log path.
        state["logs"]                  - agent execution summary.
        state["tool_calls"]            - tool-level trace records.

    Args:
        state: Shared FoodState dictionary.

    Returns:
        Updated FoodState dictionary.
    """
    print(f"\n[Agent 4] Starting {AGENT_NAME}...")

    try:
        _ensure_agent4_state(state)

        scored_meals: List[Dict[str, Any]] = state["scored_meals"]
        preferences: Dict[str, Any] = state["preferences"]
        run_id: str = str(state.get("run_id", "unknown"))

        # Step 1: Rank meals
        ranked = rank_meals(scored_meals)
        log_tool_call(
            state,
            "rank_meals",
            {"scored_meal_count": len(scored_meals)},
            {"ranked_count": len(ranked)},
        )

        # Step 2: Diversify top meals
        top_meals = diversify_meals(ranked, top_n=TOP_N)
        log_tool_call(
            state,
            "diversify_meals",
            {"ranked_count": len(ranked), "top_n": TOP_N},
            {"selected_count": len(top_meals)},
        )

        # Step 3: Add rank badges
        enriched = enrich_with_rank(top_meals)
        log_tool_call(
            state,
            "enrich_with_rank",
            {"meal_count": len(top_meals)},
            {"enriched_count": len(enriched)},
        )

        # Step 4: Add explainable selection description
        for meal in enriched:
            try:
                meal["selection_description"] = generate_selection_description(
                    meal, preferences
                )
            except Exception as exc:
                meal["selection_description"] = meal.get(
                    "reason",
                    "Recommended based on final score and user preferences.",
                )
                state["errors"].append(
                    f"Selection description fallback used for "
                    f"{meal.get('name', 'Unknown Meal')}: {exc}"
                )

        log_tool_call(
            state,
            "generate_selection_description",
            {"meal_count": len(enriched)},
            {"status": "completed_with_fallback_support"},
        )

        # Step 5: Save final recommendations to state
        state["final_recommendations"] = enriched

        # Step 6: Build Markdown report
        report_md = build_markdown_report(enriched, preferences)
        log_tool_call(
            state,
            "build_markdown_report",
            {"meal_count": len(enriched)},
            {"report_length_chars": len(report_md)},
        )

        # Step 7: Save Markdown and JSON outputs
        report_path = save_markdown_report(report_md)
        json_path = save_json_results(enriched, preferences)

        state["report_path"] = report_path
        state["json_path"] = json_path

        log_tool_call(
            state,
            "save_output_files",
            {"run_id": run_id},
            {"report_path": report_path, "json_path": json_path},
        )

        # Step 8: Log Agent 4 execution
        log_agent_step(
            state=state,
            agent_name=AGENT_NAME,
            input_data={
                "scored_meals_received": len(scored_meals),
                "top_n_requested": TOP_N,
                "preferences": preferences,
            },
            output_data={
                "recommendations_count": len(enriched),
                "top_meal_name": enriched[0].get("name") if enriched else None,
                "top_meal_score": enriched[0].get("final_score") if enriched else None,
                "report_path": report_path,
                "json_path": json_path,
                "status": "success" if enriched else "no_recommendations",
            },
        )

        # Step 9: Persist full trace
        try:
            trace_path = persist_trace(state, run_id)
            state["trace_path"] = trace_path
            print(f"[Agent 4] Trace saved -> {trace_path}")
        except Exception as exc:
            state["errors"].append(f"Trace persistence failed: {exc}")
            print(f"[Agent 4] Warning: trace persistence failed: {exc}")

        # Step 10: Print console output
        print_console_summary(enriched, preferences)
        print(f"[Agent 4] Done. {len(enriched)} recommendations generated.")

        return state

    except Exception as exc:
        state.setdefault("errors", [])
        state["errors"].append(f"{AGENT_NAME} failed: {exc}")

        log_agent_step(
            state=state,
            agent_name=AGENT_NAME,
            input_data={
                "scored_meals_received": len(state.get("scored_meals", []))
                if isinstance(state.get("scored_meals", []), list)
                else "invalid",
            },
            output_data={
                "status": "failed",
                "error": str(exc),
            },
        )

        print(f"[Agent 4] Error: {exc}")
        return state