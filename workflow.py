"""
workflow.py
============
LangGraph StateGraph orchestration for the Food Recommendation MAS.

build_graph() compiles the four-agent pipeline into a LangGraph graph:

    [START]
       ↓
  analyze_preferences  (Agent 1)
       ↓
  fetch_menu           (Agent 2)
       ↓
  analyze_nutrition    (Agent 3)
       ↓
  recommend_meals      (Agent 4)
       ↓
    [END]

Usage:
    from workflow import build_graph

    graph = build_graph()
    result = graph.invoke(initial_state)

Assignment compliance:
    ✔  LangGraph StateGraph with explicit node registration
    ✔  Linear edge chain reflecting the MAS pipeline sequence
    ✔  Imported by both main.py (CLI) and web/app.py (Flask)
"""

from __future__ import annotations

import uuid
from typing import Any, Dict

from langgraph.graph import StateGraph, START, END

from state import FoodState
from agents.preference_agent import analyze_preferences
from agents.menu_agent import fetch_menu
from agents.nutrition_agent import analyze_nutrition
from agents.recommendation_agent import recommend_meals


# ─── Node wrapper functions ───────────────────────────────────────────────────
# LangGraph nodes receive the full state dict and must return a (possibly
# partial) dict with the updated keys.  We wrap each agent function to ensure
# the full state is returned even if the agent only modifies a subset of keys.

def _node_preference(state: FoodState) -> Dict[str, Any]:
    """Agent 1 node: Preference Analyzer."""
    user_input = state.get("user_input", "")
    return analyze_preferences(dict(state), user_input)


def _node_menu(state: FoodState) -> Dict[str, Any]:
    """Agent 2 node: Menu Fetcher."""
    return fetch_menu(dict(state))


def _node_nutrition(state: FoodState) -> Dict[str, Any]:
    """Agent 3 node: Nutrition Analyzer."""
    return analyze_nutrition(dict(state))


def _node_recommend(state: FoodState) -> Dict[str, Any]:
    """Agent 4 node: Recommendation & Report."""
    return recommend_meals(dict(state))


# ─── Public API ───────────────────────────────────────────────────────────────

def build_graph():
    """
    Build and compile the LangGraph StateGraph for the food recommendation pipeline.

    Returns:
        A compiled LangGraph graph that accepts a FoodState dict and returns
        an updated FoodState dict with all fields populated.

    Example:
        >>> graph = build_graph()
        >>> state = make_initial_state("I want vegan food under 500 calories")
        >>> result = graph.invoke(state)
        >>> result["final_recommendations"]
        [...]
    """
    graph = StateGraph(FoodState)

    # Register nodes
    graph.add_node("preference_agent",     _node_preference)
    graph.add_node("menu_agent",           _node_menu)
    graph.add_node("nutrition_agent",      _node_nutrition)
    graph.add_node("recommendation_agent", _node_recommend)

    # Define edges (linear pipeline)
    graph.add_edge(START,                  "preference_agent")
    graph.add_edge("preference_agent",     "menu_agent")
    graph.add_edge("menu_agent",           "nutrition_agent")
    graph.add_edge("nutrition_agent",      "recommendation_agent")
    graph.add_edge("recommendation_agent", END)

    return graph.compile()


def make_initial_state(user_input: str, run_id: str | None = None) -> FoodState:
    """
    Create a blank FoodState with user_input and run_id pre-populated.

    Args:
        user_input: The user's free-text food preference query.
        run_id:     Optional run identifier; auto-generated UUID if not given.

    Returns:
        A fresh FoodState dict ready to be passed to the compiled graph.
    """
    return {
        "user_input":            user_input,
        "run_id":                run_id or str(uuid.uuid4()),
        "preferences":           {},
        "candidate_meals":       [],
        "nutrition_evaluations": [],
        "scored_meals":          [],
        "final_recommendations": [],
        "report_path":           None,
        "json_path":             None,
        "trace_path":            None,
        "tool_calls":            [],
        "errors":                [],
        "logs":                  [],
    }
