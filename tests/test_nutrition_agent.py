"""
tests/test_nutrition_agent.py
==============================
Comprehensive tests for Agent 3 (NutritionAnalyzerAgent) and its tool
(tools/nutrition_tool.py).

Test categories:
    1. Unit — nutrition_tool.evaluate_meal_nutrition (per-meal evaluation)
    2. Unit — nutrition_tool.batch_evaluate
    3. Unit — nutrition_tool.compute_health_flags
    4. Integration — agents/nutrition_agent.analyze_nutrition (full agent)
    5. Edge cases — missing fields, invalid types, empty lists
    6. Diet compliance — vegan, vegetarian, keto rejection paths

Run with:
    pytest tests/test_nutrition_agent.py -v
"""

from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.nutrition_tool import (
    evaluate_meal_nutrition,
    batch_evaluate,
    compute_health_flags,
)
from agents.nutrition_agent import analyze_nutrition


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _meal(
    name="Tofu Bowl", calories=300.0, protein=20.0, fat=10.0, carbs=35.0,
    diet_type="vegan", cuisine="Asian", allergens="none",
    ingredients="tofu|rice|soy", sugar=None, sodium=None,
):
    m = dict(name=name, calories=calories, protein=protein, fat=fat,
             carbs=carbs, diet_type=diet_type, cuisine=cuisine,
             allergens=allergens, ingredients=ingredients)
    if sugar  is not None: m["sugar"]  = sugar
    if sodium is not None: m["sodium"] = sodium
    return m


@pytest.fixture
def vegan_meal():
    return _meal()

@pytest.fixture
def chicken_meal():
    return _meal(name="Chicken Rice", diet_type="", ingredients="chicken|rice", calories=400)

@pytest.fixture
def vegan_prefs():
    return {"diet": "vegan", "calorie_limit": 500, "exclude": [], "cuisine": None}

@pytest.fixture
def base_state():
    return {
        "user_input": "", "run_id": "test-001",
        "preferences": {}, "candidate_meals": [],
        "nutrition_evaluations": [], "scored_meals": [],
        "final_recommendations": [], "report_path": None,
        "json_path": None, "tool_calls": [], "errors": [], "logs": [],
    }


# ═════════════════════════════════════════════════════════════════════════════
# 1. Unit — evaluate_meal_nutrition
# ═════════════════════════════════════════════════════════════════════════════

class TestEvaluateMealNutrition:

    def test_accepts_valid_vegan_meal(self, vegan_meal, vegan_prefs):
        result = evaluate_meal_nutrition(vegan_meal, vegan_prefs)
        assert result["rejected"] is False
        assert 0.0 <= result["final_score"] <= 1.0

    def test_rejects_meal_over_calorie_limit(self, vegan_prefs):
        meal = _meal(calories=600.0)
        result = evaluate_meal_nutrition(meal, vegan_prefs)
        assert result["rejected"] is True
        assert "calorie" in result["rejection_reason"].lower()

    def test_rejects_meal_missing_calories(self, vegan_prefs):
        meal = _meal()
        del meal["calories"]
        result = evaluate_meal_nutrition(meal, vegan_prefs)
        assert result["rejected"] is True
        assert "calories" in result["rejection_reason"]

    def test_rejects_meal_missing_protein(self, vegan_prefs):
        meal = _meal()
        del meal["protein"]
        result = evaluate_meal_nutrition(meal, vegan_prefs)
        assert result["rejected"] is True

    def test_rejects_vegan_meal_with_chicken(self, chicken_meal, vegan_prefs):
        result = evaluate_meal_nutrition(chicken_meal, vegan_prefs)
        assert result["rejected"] is True
        assert "vegan" in result["rejection_reason"].lower()

    def test_rejects_excluded_ingredient(self, vegan_meal):
        prefs = {"diet": None, "calorie_limit": None, "exclude": ["tofu"]}
        result = evaluate_meal_nutrition(vegan_meal, prefs)
        assert result["rejected"] is True
        assert "tofu" in result["rejection_reason"]

    def test_keto_rejects_high_carb(self):
        meal = _meal(carbs=80.0, diet_type="keto")
        prefs = {"diet": "keto", "calorie_limit": None, "exclude": []}
        result = evaluate_meal_nutrition(meal, prefs)
        assert result["rejected"] is True
        assert "keto" in result["rejection_reason"].lower()

    def test_keto_accepts_low_carb(self):
        meal = _meal(carbs=15.0, fat=20.0, diet_type="keto")
        prefs = {"diet": "keto", "calorie_limit": None, "exclude": []}
        result = evaluate_meal_nutrition(meal, prefs)
        assert result["rejected"] is False

    def test_accepted_meal_has_required_keys(self, vegan_meal, vegan_prefs):
        result = evaluate_meal_nutrition(vegan_meal, vegan_prefs)
        for key in ["meal", "rejected", "nutrition_score", "diet_match",
                    "final_score", "category", "health_flags", "reason"]:
            assert key in result, f"Missing key: {key}"

    def test_category_is_valid_value(self, vegan_meal, vegan_prefs):
        result = evaluate_meal_nutrition(vegan_meal, vegan_prefs)
        assert result["category"] in ("healthy", "moderate", "unhealthy")

    def test_raises_type_error_on_non_mapping_meal(self, vegan_prefs):
        with pytest.raises(TypeError, match="meal"):
            evaluate_meal_nutrition("not a dict", vegan_prefs)

    def test_raises_type_error_on_non_mapping_prefs(self, vegan_meal):
        with pytest.raises(TypeError, match="preferences"):
            evaluate_meal_nutrition(vegan_meal, "not a dict")

    def test_vegetarian_rejects_meat(self):
        meal = _meal(ingredients="lamb|potato", diet_type="")
        prefs = {"diet": "vegetarian", "calorie_limit": None, "exclude": []}
        result = evaluate_meal_nutrition(meal, prefs)
        assert result["rejected"] is True

    def test_no_preferences_accepts_any_valid_meal(self):
        meal = _meal()
        result = evaluate_meal_nutrition(meal, {})
        assert result["rejected"] is False

    def test_final_score_between_0_and_1(self, vegan_meal, vegan_prefs):
        result = evaluate_meal_nutrition(vegan_meal, vegan_prefs)
        assert 0.0 <= result["final_score"] <= 1.0

    def test_nutrition_score_between_0_and_1(self, vegan_meal, vegan_prefs):
        result = evaluate_meal_nutrition(vegan_meal, vegan_prefs)
        assert 0.0 <= result["nutrition_score"] <= 1.0

    def test_high_protein_goal_boosts_score(self):
        meal = _meal(protein=35.0, calories=400.0)
        prefs_no_goal  = {"diet": None, "calorie_limit": 500, "exclude": [], "goal": None}
        prefs_hp_goal  = {"diet": None, "calorie_limit": 500, "exclude": [], "goal": "high protein"}
        r_base = evaluate_meal_nutrition(meal, prefs_no_goal)
        r_hp   = evaluate_meal_nutrition(meal, prefs_hp_goal)
        assert r_hp["nutrition_score"] >= r_base["nutrition_score"]


# ═════════════════════════════════════════════════════════════════════════════
# 2. Unit — batch_evaluate
# ═════════════════════════════════════════════════════════════════════════════

class TestBatchEvaluate:

    def test_returns_tuple_of_two_lists(self, vegan_prefs):
        result = batch_evaluate([_meal()], vegan_prefs)
        assert isinstance(result, tuple) and len(result) == 2
        all_evals, accepted = result
        assert isinstance(all_evals, list) and isinstance(accepted, list)

    def test_accepted_subset_of_all(self, vegan_prefs):
        meals = [_meal(), _meal(name="Chicken", ingredients="chicken", diet_type="")]
        all_evals, accepted = batch_evaluate(meals, vegan_prefs)
        assert len(accepted) <= len(all_evals)

    def test_empty_input_returns_empty_lists(self, vegan_prefs):
        all_evals, accepted = batch_evaluate([], vegan_prefs)
        assert all_evals == [] and accepted == []

    def test_invalid_meal_record_does_not_crash(self, vegan_prefs):
        all_evals, accepted = batch_evaluate(["not a dict"], vegan_prefs)
        assert len(all_evals) == 1
        assert all_evals[0]["rejected"] is True

    def test_raises_type_error_on_non_list(self, vegan_prefs):
        with pytest.raises(TypeError):
            batch_evaluate("not a list", vegan_prefs)

    def test_all_valid_meals_all_accepted_with_no_prefs(self):
        meals = [_meal(), _meal(name="Quinoa Bowl", diet_type="")]
        all_evals, accepted = batch_evaluate(meals, {})
        assert len(accepted) == len(all_evals) == 2


# ═════════════════════════════════════════════════════════════════════════════
# 3. Unit — compute_health_flags
# ═════════════════════════════════════════════════════════════════════════════

class TestComputeHealthFlags:

    def test_high_sugar_flag(self):
        flags = compute_health_flags(300, 20, 30, 10, sugar=30.0, sodium=None, calorie_limit=None)
        assert "high_sugar" in flags

    def test_high_sodium_flag(self):
        flags = compute_health_flags(300, 20, 30, 10, sugar=None, sodium=800.0, calorie_limit=None)
        assert "high_sodium" in flags

    def test_high_fat_flag(self):
        flags = compute_health_flags(300, 20, 30, fat=35.0, sugar=None, sodium=None, calorie_limit=None)
        assert "high_fat" in flags

    def test_low_protein_flag(self):
        flags = compute_health_flags(300, protein=5.0, carbs=30, fat=10, sugar=None, sodium=None, calorie_limit=None)
        assert "low_protein" in flags

    def test_high_carb_flag(self):
        flags = compute_health_flags(300, 20, carbs=100.0, fat=10, sugar=None, sodium=None, calorie_limit=None)
        assert "high_carb" in flags

    def test_no_flags_for_healthy_meal(self):
        flags = compute_health_flags(300, 22, 40, 12, sugar=10.0, sodium=400.0, calorie_limit=500)
        assert flags == []


# ═════════════════════════════════════════════════════════════════════════════
# 4. Integration — analyze_nutrition (full agent)
# ═════════════════════════════════════════════════════════════════════════════

class TestAnalyzeNutrition:

    def test_writes_scored_meals_to_state(self, base_state):
        base_state["candidate_meals"] = [_meal()]
        base_state["preferences"] = {"diet": "vegan", "calorie_limit": 500, "exclude": []}
        result = analyze_nutrition(base_state)
        assert "scored_meals" in result

    def test_writes_nutrition_evaluations_to_state(self, base_state):
        base_state["candidate_meals"] = [_meal()]
        base_state["preferences"] = {"diet": "vegan", "calorie_limit": 500, "exclude": []}
        result = analyze_nutrition(base_state)
        assert "nutrition_evaluations" in result
        assert len(result["nutrition_evaluations"]) == 1

    def test_rejected_meal_not_in_scored_meals(self, base_state):
        base_state["candidate_meals"] = [_meal(calories=9999)]
        base_state["preferences"] = {"diet": None, "calorie_limit": 500, "exclude": []}
        result = analyze_nutrition(base_state)
        assert len(result["scored_meals"]) == 0

    def test_log_entry_appended(self, base_state):
        base_state["candidate_meals"] = [_meal()]
        base_state["preferences"] = {"diet": "vegan", "calorie_limit": 500, "exclude": []}
        result = analyze_nutrition(base_state)
        agents = [l["agent"] for l in result["logs"]]
        assert "NutritionAnalyzerAgent" in agents

    def test_tool_call_logged(self, base_state):
        base_state["candidate_meals"] = [_meal()]
        base_state["preferences"] = {"diet": "vegan", "calorie_limit": 500, "exclude": []}
        result = analyze_nutrition(base_state)
        tools = [t["tool"] for t in result["tool_calls"]]
        assert "batch_evaluate" in tools

    def test_empty_candidate_meals_returns_empty_scored(self, base_state):
        base_state["candidate_meals"] = []
        base_state["preferences"] = {"diet": "vegan", "calorie_limit": 500, "exclude": []}
        result = analyze_nutrition(base_state)
        assert result["scored_meals"] == []
        assert result["nutrition_evaluations"] == []

    def test_multiple_meals_mixed_acceptance(self, base_state):
        meals = [_meal(), _meal(name="Beef Burger", ingredients="beef", calories=700)]
        base_state["candidate_meals"] = meals
        base_state["preferences"] = {"diet": "vegan", "calorie_limit": 500, "exclude": []}
        result = analyze_nutrition(base_state)
        assert len(result["nutrition_evaluations"]) == 2
        assert len(result["scored_meals"]) < 2  # beef burger should be rejected
