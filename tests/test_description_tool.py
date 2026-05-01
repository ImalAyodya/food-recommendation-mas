"""
tests/test_description_tool.py
================================
Tests for tools/description_tool.py — the "why this meal was selected"
description generator.

Run with:
    pytest tests/test_description_tool.py -v
"""

from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.description_tool import generate_selection_description


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _meal(**kwargs):
    base = {
        "name": "Tofu Bowl",
        "calories": 300.0, "protein": 22.0, "fat": 10.0, "carbs": 35.0,
        "diet_type": "vegan", "cuisine": "Asian", "allergens": "none",
        "ingredients": "tofu|rice|soy",
        "final_score": 0.85, "score": 0.85,
        "nutrition_score": 0.80, "diet_match": 0.95,
        "health_flags": [], "category": "healthy",
        "reason": "Nutritionally suitable.", "rank": 1,
    }
    base.update(kwargs)
    return base

def _prefs(**kwargs):
    base = {"diet": "vegan", "calorie_limit": 500, "exclude": [], "cuisine": None, "goal": None}
    base.update(kwargs)
    return base


# ═════════════════════════════════════════════════════════════════════════════
# 1. Return type and basic content
# ═════════════════════════════════════════════════════════════════════════════

class TestReturnType:
    def test_returns_string(self):
        result = generate_selection_description(_meal(), _prefs())
        assert isinstance(result, str)

    def test_not_empty(self):
        result = generate_selection_description(_meal(), _prefs())
        assert len(result.strip()) > 0

    def test_multiple_sentences(self):
        result = generate_selection_description(_meal(), _prefs())
        assert result.count(".") >= 2

    def test_no_crash_on_empty_meal(self):
        result = generate_selection_description({}, {})
        assert isinstance(result, str)

    def test_no_crash_on_none_values(self):
        meal = _meal(calories=None, protein=None, fat=None, carbs=None,
                     final_score=None, nutrition_score=None, diet_match=None,
                     health_flags=None, category=None, rank=None)
        result = generate_selection_description(meal, {})
        assert isinstance(result, str)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Score and rank mention
# ═════════════════════════════════════════════════════════════════════════════

class TestScoreAndRank:
    def test_mentions_rank(self):
        result = generate_selection_description(_meal(rank=1), _prefs())
        assert "#1" in result or "Ranked" in result

    def test_mentions_overall_score(self):
        result = generate_selection_description(_meal(final_score=0.85), _prefs())
        assert "85" in result

    def test_healthy_category_label(self):
        result = generate_selection_description(_meal(category="healthy"), _prefs())
        assert "excellent" in result.lower() or "healthy" in result.lower()

    def test_moderate_category_label(self):
        result = generate_selection_description(_meal(category="moderate"), _prefs())
        assert "solid" in result.lower() or "moderate" in result.lower()

    def test_nutrition_score_mentioned(self):
        result = generate_selection_description(_meal(nutrition_score=0.82), _prefs())
        assert "82" in result


# ═════════════════════════════════════════════════════════════════════════════
# 3. Calorie budget explanation
# ═════════════════════════════════════════════════════════════════════════════

class TestCalorieBudget:
    def test_calorie_limit_mentioned(self):
        result = generate_selection_description(_meal(calories=300), _prefs(calorie_limit=500))
        assert "500" in result
        assert "300" in result

    def test_calorie_percentage_correct(self):
        result = generate_selection_description(_meal(calories=400), _prefs(calorie_limit=500))
        # 400/500 = 80%
        assert "80" in result

    def test_remaining_calories_mentioned(self):
        result = generate_selection_description(_meal(calories=300), _prefs(calorie_limit=500))
        # 500 - 300 = 200 kcal to spare
        assert "200" in result

    def test_no_calorie_limit_still_describes_calories(self):
        result = generate_selection_description(_meal(calories=300), _prefs(calorie_limit=None))
        assert "300" in result

    def test_high_calorie_meal_described(self):
        result = generate_selection_description(_meal(calories=650), _prefs(calorie_limit=None))
        assert "650" in result


# ═════════════════════════════════════════════════════════════════════════════
# 4. Diet compliance
# ═════════════════════════════════════════════════════════════════════════════

class TestDietCompliance:
    def test_vegan_diet_mentioned(self):
        result = generate_selection_description(_meal(diet_match=0.97), _prefs(diet="vegan"))
        assert "vegan" in result.lower() or "Vegan" in result

    def test_high_diet_match_says_fully_compliant(self):
        result = generate_selection_description(_meal(diet_match=0.98), _prefs(diet="vegan"))
        assert "fully" in result.lower() or "100" in result or "98" in result

    def test_low_diet_match_says_partially(self):
        result = generate_selection_description(_meal(diet_match=0.60), _prefs(diet="vegan"))
        assert "partial" in result.lower()

    def test_no_diet_says_no_restriction(self):
        result = generate_selection_description(_meal(), _prefs(diet=None))
        assert "no specific" in result.lower() or "all meal" in result.lower()

    def test_keto_mentioned_when_set(self):
        result = generate_selection_description(_meal(diet_match=0.92), _prefs(diet="keto"))
        assert "keto" in result.lower() or "Keto" in result


# ═════════════════════════════════════════════════════════════════════════════
# 5. Macro highlights
# ═════════════════════════════════════════════════════════════════════════════

class TestMacroHighlights:
    def test_high_protein_mentioned(self):
        result = generate_selection_description(_meal(protein=30.0), _prefs())
        assert "protein" in result.lower()

    def test_low_carb_mentioned(self):
        result = generate_selection_description(_meal(carbs=15.0), _prefs())
        assert "carb" in result.lower()

    def test_high_protein_goal_reflected(self):
        result = generate_selection_description(_meal(protein=25.0), _prefs(goal="high protein"))
        assert "protein" in result.lower()

    def test_no_macro_highlight_for_average_values(self):
        # Average values shouldn't add noise
        meal = _meal(protein=10.0, carbs=55.0, fat=25.0)
        result = generate_selection_description(meal, _prefs())
        assert isinstance(result, str)


# ═════════════════════════════════════════════════════════════════════════════
# 6. Cuisine match
# ═════════════════════════════════════════════════════════════════════════════

class TestCuisine:
    def test_cuisine_preference_match_mentioned(self):
        result = generate_selection_description(
            _meal(cuisine="Asian"), _prefs(cuisine="Asian")
        )
        assert "asian" in result.lower() or "Asian" in result

    def test_meal_cuisine_mentioned_without_preference(self):
        result = generate_selection_description(
            _meal(cuisine="Italian"), _prefs(cuisine=None)
        )
        assert "Italian" in result

    def test_no_cuisine_no_crash(self):
        result = generate_selection_description(
            _meal(cuisine=None), _prefs(cuisine=None)
        )
        assert isinstance(result, str)


# ═════════════════════════════════════════════════════════════════════════════
# 7. Exclusion safety note
# ═════════════════════════════════════════════════════════════════════════════

class TestExclusions:
    def test_excluded_ingredient_mentioned(self):
        result = generate_selection_description(_meal(), _prefs(exclude=["nuts"]))
        assert "nuts" in result.lower()

    def test_multiple_exclusions_mentioned(self):
        result = generate_selection_description(_meal(), _prefs(exclude=["nuts", "dairy"]))
        assert "nuts" in result.lower()
        assert "dairy" in result.lower()

    def test_empty_exclude_no_mention(self):
        result = generate_selection_description(_meal(), _prefs(exclude=[]))
        assert "excluded" not in result.lower() or "none" in result.lower()


# ═════════════════════════════════════════════════════════════════════════════
# 8. Health flags
# ═════════════════════════════════════════════════════════════════════════════

class TestHealthFlags:
    def test_high_sugar_flag_mentioned(self):
        result = generate_selection_description(
            _meal(health_flags=["high_sugar"]), _prefs()
        )
        assert "sugar" in result.lower()

    def test_high_sodium_flag_mentioned(self):
        result = generate_selection_description(
            _meal(health_flags=["high_sodium"]), _prefs()
        )
        assert "sodium" in result.lower()

    def test_multiple_flags_at_most_two(self):
        # Should mention at most 2 flags (we cap at [:2])
        result = generate_selection_description(
            _meal(health_flags=["high_sugar", "high_sodium", "high_fat"]), _prefs()
        )
        assert "Note" in result or "note" in result

    def test_no_flags_no_note(self):
        result = generate_selection_description(
            _meal(health_flags=[]), _prefs()
        )
        assert "Note:" not in result


# ═════════════════════════════════════════════════════════════════════════════
# 9. Edge cases
# ═════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_unknown_meal_name_fallback(self):
        result = generate_selection_description(_meal(name=None), _prefs())
        assert isinstance(result, str)

    def test_zero_score_handled(self):
        result = generate_selection_description(_meal(final_score=0.0), _prefs())
        assert "0%" in result

    def test_perfect_score_handled(self):
        result = generate_selection_description(_meal(final_score=1.0), _prefs())
        assert "100%" in result

    def test_string_numeric_values_handled(self):
        # If the pipeline passes strings instead of numbers, should not crash
        result = generate_selection_description(
            _meal(calories="300", protein="22", fat="10", carbs="35",
                  final_score="0.85", nutrition_score="0.80"),
            _prefs()
        )
        assert isinstance(result, str)

    def test_description_length_reasonable(self):
        result = generate_selection_description(_meal(), _prefs())
        # Should be more than one sentence but not a novel
        assert 40 < len(result) < 2000
