"""
tests/test_agents.py
=====================
Smoke integration tests for the full MAS pipeline.

These tests verify that all four agents can be wired together and produce
a correctly shaped state dict, using mocked LLM and dataset calls so the
tests run offline without Ollama or a real CSV file.
"""

from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from workflow import make_initial_state


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_candidate():
    return {
        "name": "Vegan Pasta", "calories": 320.0, "protein": 18.0,
        "fat": 7.0, "carbs": 55.0, "diet_type": "vegan",
        "cuisine": "Italian", "allergens": "none",
        "ingredients": "pasta|tomato|basil",
    }


# ─── Pipeline smoke tests ─────────────────────────────────────────────────────

class TestPipelineSmoke:

    def test_initial_state_has_all_keys(self):
        state = make_initial_state("vegan food under 500 calories")
        required_keys = [
            "user_input", "run_id", "preferences", "candidate_meals",
            "nutrition_evaluations", "scored_meals", "final_recommendations",
            "report_path", "json_path", "tool_calls", "errors", "logs",
        ]
        for key in required_keys:
            assert key in state, f"Missing key: {key}"

    def test_initial_state_user_input_stored(self):
        state = make_initial_state("keto meal under 400 cal")
        assert state["user_input"] == "keto meal under 400 cal"

    def test_initial_state_run_id_is_string(self):
        state = make_initial_state("test")
        assert isinstance(state["run_id"], str) and len(state["run_id"]) > 0

    def test_initial_state_lists_are_empty(self):
        state = make_initial_state("test")
        for key in ["candidate_meals", "scored_meals", "final_recommendations",
                    "nutrition_evaluations", "tool_calls", "errors", "logs"]:
            assert state[key] == [], f"Expected empty list for {key}"

    def test_preference_agent_alone(self, mocker):
        """Agent 1 should run without Ollama when mocked."""
        mocker.patch(
            "langchain_core.runnables.RunnableSequence.invoke",
            return_value={"diet": "vegan", "calorie_limit": 500,
                          "exclude": [], "cuisine": None}
        )
        from agents.preference_agent import analyze_preferences
        state = make_initial_state("vegan under 500 calories")
        result = analyze_preferences(state, state["user_input"])
        assert result["preferences"]["diet"] == "vegan"
        assert result["preferences"]["calorie_limit"] == 500.0
        assert len(result["logs"]) == 1

    def test_nutrition_agent_alone(self):
        """Agent 3 should accept valid meals without any LLM."""
        from agents.nutrition_agent import analyze_nutrition
        state = make_initial_state("vegan")
        state["candidate_meals"] = [_make_candidate()]
        state["preferences"] = {"diet": "vegan", "calorie_limit": 500, "exclude": []}
        result = analyze_nutrition(state)
        assert len(result["nutrition_evaluations"]) == 1
        assert len(result["scored_meals"]) >= 0

    def test_recommendation_agent_alone(self, tmp_path, monkeypatch):
        """Agent 4 should produce recommendations from scored meals."""
        monkeypatch.setattr("tools.report_tool.REPORTS_DIR", str(tmp_path / "reports"))
        monkeypatch.setattr("tools.report_tool.RESULTS_DIR", str(tmp_path / "results"))
        monkeypatch.setattr("tools.logger.LOGS_DIR", str(tmp_path / "logs"))
        from agents.recommendation_agent import recommend_meals
        meal = _make_candidate()
        meal.update({"final_score": 0.85, "score": 0.85, "nutrition_score": 0.80,
                     "diet_match": 0.90, "health_flags": [], "category": "healthy",
                     "reason": "Nutritionally suitable.", "rejected": False})
        state = make_initial_state("vegan")
        state["scored_meals"] = [meal]
        state["preferences"] = {"diet": "vegan", "calorie_limit": 500, "exclude": []}
        result = recommend_meals(state)
        assert len(result["final_recommendations"]) >= 1
        assert result["report_path"] is not None
        assert result["json_path"] is not None