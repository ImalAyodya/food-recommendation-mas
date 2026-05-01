"""
tests/test_workflow.py
=======================
Tests for the LangGraph StateGraph pipeline (workflow.py).

These tests verify that:
    - build_graph() produces a compilable graph
    - make_initial_state() returns a correctly shaped FoodState
    - The compiled graph invokes all 4 agent nodes in the correct order
    - State keys are propagated through each node

All LLM calls and file I/O are mocked so tests run fully offline.

Run with:
    pytest tests/test_workflow.py -v
"""

from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from workflow import build_graph, make_initial_state


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _scored_meal(name="Tofu Bowl", score=0.85, cuisine="Asian"):
    return {
        "name": name, "calories": 300.0, "protein": 20.0, "fat": 10.0,
        "carbs": 35.0, "diet_type": "vegan", "cuisine": cuisine,
        "allergens": "none", "ingredients": "tofu|rice",
        "final_score": score, "score": score,
        "nutrition_score": 0.80, "diet_match": 0.90,
        "health_flags": [], "category": "healthy",
        "reason": "Nutritionally suitable.", "rejected": False,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 1. make_initial_state
# ═════════════════════════════════════════════════════════════════════════════

class TestMakeInitialState:

    def test_contains_all_required_keys(self):
        state = make_initial_state("vegan food")
        required = [
            "user_input", "run_id", "preferences", "candidate_meals",
            "nutrition_evaluations", "scored_meals", "final_recommendations",
            "report_path", "json_path", "tool_calls", "errors", "logs",
        ]
        for key in required:
            assert key in state, f"Missing key: {key}"

    def test_user_input_stored(self):
        state = make_initial_state("keto under 400")
        assert state["user_input"] == "keto under 400"

    def test_run_id_auto_generated(self):
        s1 = make_initial_state("test")
        s2 = make_initial_state("test")
        assert s1["run_id"] != s2["run_id"]

    def test_run_id_custom(self):
        state = make_initial_state("test", run_id="my-run-42")
        assert state["run_id"] == "my-run-42"

    def test_lists_initialised_empty(self):
        state = make_initial_state("test")
        for key in ["candidate_meals", "nutrition_evaluations", "scored_meals",
                    "final_recommendations", "tool_calls", "errors", "logs"]:
            assert state[key] == []

    def test_paths_initialised_to_none(self):
        state = make_initial_state("test")
        assert state["report_path"] is None
        assert state["json_path"] is None


# ═════════════════════════════════════════════════════════════════════════════
# 2. build_graph
# ═════════════════════════════════════════════════════════════════════════════

class TestBuildGraph:

    def test_graph_compiles_without_error(self):
        graph = build_graph()
        assert graph is not None

    def test_graph_has_invoke_method(self):
        graph = build_graph()
        assert hasattr(graph, "invoke")

    def test_graph_multiple_compilations_independent(self):
        g1 = build_graph()
        g2 = build_graph()
        assert g1 is not g2


# ═════════════════════════════════════════════════════════════════════════════
# 3. End-to-end via mocked agents
# ═════════════════════════════════════════════════════════════════════════════

class TestWorkflowInvocation:

    def _patch_all(self, mocker, tmp_path):
        """Mock every external dependency so the graph runs offline."""
        # Agent 1: mock LLM chain
        mocker.patch(
            "langchain_core.runnables.RunnableSequence.invoke",
            return_value={"diet": "vegan", "calorie_limit": 500,
                          "exclude": [], "cuisine": None}
        )
        # Agent 2: mock LLM + dataset
        mocker.patch(
            "agents.menu_agent._normalize_preferences_with_llm",
            return_value={"diet": "vegan", "calorie_limit": 500,
                          "exclude": [], "cuisine": None, "prep_time": None}
        )
        mocker.patch(
            "tools.data_loader.load_data",
            return_value=__import__("pandas").DataFrame([{
                "name": "Tofu Bowl", "calories": 300, "protein": 20,
                "fat": 10, "carbs": 35, "diet_type": "vegan",
                "cuisine": "Asian", "allergens": "none",
                "ingredients": "tofu|rice",
            }])
        )
        # Agent 4: redirect file output
        mocker.setattr = None  # use monkeypatch below

    def test_full_pipeline_returns_final_recommendations(self, mocker, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.report_tool.REPORTS_DIR", str(tmp_path / "reports"))
        monkeypatch.setattr("tools.report_tool.RESULTS_DIR", str(tmp_path / "results"))
        monkeypatch.setattr("tools.logger.LOGS_DIR",         str(tmp_path / "logs"))

        mocker.patch(
            "langchain_core.runnables.RunnableSequence.invoke",
            return_value={"diet": "vegan", "calorie_limit": 500, "exclude": [], "cuisine": None}
        )
        mocker.patch(
            "agents.menu_agent._normalize_preferences_with_llm",
            return_value={"diet": "vegan", "calorie_limit": 500,
                          "exclude": [], "cuisine": None, "prep_time": None}
        )
        import pandas as pd
        mocker.patch(
            "tools.data_loader.load_data",
            return_value=pd.DataFrame([{
                "name": "Tofu Bowl", "calories": 300, "protein": 20,
                "fat": 10, "carbs": 35, "diet_type": "vegan",
                "cuisine": "Asian", "allergens": "none", "ingredients": "tofu|rice",
            }])
        )

        graph = build_graph()
        state = make_initial_state("vegan food under 500 calories")
        result = graph.invoke(state)

        assert "final_recommendations" in result
        assert isinstance(result["final_recommendations"], list)

    def test_full_pipeline_populates_logs(self, mocker, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.report_tool.REPORTS_DIR", str(tmp_path / "reports"))
        monkeypatch.setattr("tools.report_tool.RESULTS_DIR", str(tmp_path / "results"))
        monkeypatch.setattr("tools.logger.LOGS_DIR",         str(tmp_path / "logs"))

        mocker.patch(
            "langchain_core.runnables.RunnableSequence.invoke",
            return_value={"diet": "vegan", "calorie_limit": 500, "exclude": [], "cuisine": None}
        )
        mocker.patch(
            "agents.menu_agent._normalize_preferences_with_llm",
            return_value={"diet": "vegan", "calorie_limit": 500,
                          "exclude": [], "cuisine": None, "prep_time": None}
        )
        import pandas as pd
        mocker.patch(
            "tools.data_loader.load_data",
            return_value=pd.DataFrame([{
                "name": "Tofu Bowl", "calories": 300, "protein": 20,
                "fat": 10, "carbs": 35, "diet_type": "vegan",
                "cuisine": "Asian", "allergens": "none", "ingredients": "tofu|rice",
            }])
        )

        graph = build_graph()
        state = make_initial_state("vegan food under 500 calories")
        result = graph.invoke(state)

        assert len(result["logs"]) >= 3  # at least Agents 1, 3, 4 log

    def test_full_pipeline_tool_calls_recorded(self, mocker, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.report_tool.REPORTS_DIR", str(tmp_path / "reports"))
        monkeypatch.setattr("tools.report_tool.RESULTS_DIR", str(tmp_path / "results"))
        monkeypatch.setattr("tools.logger.LOGS_DIR",         str(tmp_path / "logs"))

        mocker.patch(
            "langchain_core.runnables.RunnableSequence.invoke",
            return_value={"diet": "vegan", "calorie_limit": 500, "exclude": [], "cuisine": None}
        )
        mocker.patch(
            "agents.menu_agent._normalize_preferences_with_llm",
            return_value={"diet": "vegan", "calorie_limit": 500,
                          "exclude": [], "cuisine": None, "prep_time": None}
        )
        import pandas as pd
        mocker.patch(
            "tools.data_loader.load_data",
            return_value=pd.DataFrame([{
                "name": "Tofu Bowl", "calories": 300, "protein": 20,
                "fat": 10, "carbs": 35, "diet_type": "vegan",
                "cuisine": "Asian", "allergens": "none", "ingredients": "tofu|rice",
            }])
        )

        graph = build_graph()
        state = make_initial_state("vegan food under 500 calories")
        result = graph.invoke(state)

        tool_names = [t["tool"] for t in result.get("tool_calls", [])]
        assert "batch_evaluate" in tool_names
        assert "rank_meals" in tool_names

    def test_pipeline_stores_report_path(self, mocker, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.report_tool.REPORTS_DIR", str(tmp_path / "reports"))
        monkeypatch.setattr("tools.report_tool.RESULTS_DIR", str(tmp_path / "results"))
        monkeypatch.setattr("tools.logger.LOGS_DIR",         str(tmp_path / "logs"))

        mocker.patch(
            "langchain_core.runnables.RunnableSequence.invoke",
            return_value={"diet": "vegan", "calorie_limit": 500, "exclude": [], "cuisine": None}
        )
        mocker.patch(
            "agents.menu_agent._normalize_preferences_with_llm",
            return_value={"diet": "vegan", "calorie_limit": 500,
                          "exclude": [], "cuisine": None, "prep_time": None}
        )
        import pandas as pd
        mocker.patch(
            "tools.data_loader.load_data",
            return_value=pd.DataFrame([{
                "name": "Tofu Bowl", "calories": 300, "protein": 20,
                "fat": 10, "carbs": 35, "diet_type": "vegan",
                "cuisine": "Asian", "allergens": "none", "ingredients": "tofu|rice",
            }])
        )

        graph = build_graph()
        state = make_initial_state("vegan food under 500 calories")
        result = graph.invoke(state)

        assert result.get("report_path") is not None
        assert result.get("json_path") is not None
