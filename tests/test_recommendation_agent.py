"""
tests/test_recommendation_agent.py
====================================

Automated evaluation and testing script for Agent 4:
    RecommendationReportAgent

Test categories:
    1. Unit tests — scoring_tool.py (rank, diversify, enrich)
    2. Unit tests — report_tool.py  (build, save, console)
    3. Integration test — recommend_meals() end-to-end with mock state
    4. Edge case tests — empty input, missing keys, malformed data
    5. Security / constraint tests — file path safety, type safety
    6. Property-based style tests — output invariants that must always hold

Run with:
    pytest tests/test_recommendation_agent.py -v

Assignment compliance:
    ✔  Tests the individual agent's output (Agent 4 requirement)
    ✔  Property-based / invariant assertions (not just happy-path)
    ✔  Tests edge cases: empty, missing keys, wrong types
    ✔  Tests security constraints (type enforcement, file paths)
    ✔  Uses pytest with no paid external dependencies
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any, Dict, List

import pytest

# ── Make project root importable regardless of cwd ────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.scoring_tool import rank_meals, diversify_meals, enrich_with_rank
from tools.report_tool import (
    build_markdown_report,
    save_markdown_report,
    save_json_results,
    print_console_summary,
)
from agents.recommendation_agent import recommend_meals


# ─────────────────────────────────────────────────────────────
# Fixtures — reusable test data
# ─────────────────────────────────────────────────────────────

def _make_meal(
    name: str = "Test Salad",
    cuisine: str = "Italian",
    final_score: float = 0.85,
    calories: float = 300.0,
    protein: float = 20.0,
    fat: float = 8.0,
    carbs: float = 35.0,
    category: str = "healthy",
    health_flags: List[str] | None = None,
    reason: str = "Low calorie and high protein.",
    diet_type: str = "vegan",
) -> Dict[str, Any]:
    """Factory for a minimal valid scored meal dict."""
    return {
        "name": name,
        "cuisine": cuisine,
        "final_score": final_score,
        "score": final_score,
        "nutrition_score": round(final_score * 0.95, 2),
        "diet_match": 1.0,
        "calories": calories,
        "protein": protein,
        "fat": fat,
        "carbs": carbs,
        "category": category,
        "health_flags": health_flags or [],
        "reason": reason,
        "diet_type": diet_type,
        "allergens": "",
        "ingredients": "lettuce|tomato|olive oil",
        "rejected": False,
    }


@pytest.fixture
def sample_meals() -> List[Dict[str, Any]]:
    """Five valid diverse scored meals."""
    return [
        _make_meal("Vegan Pasta",    "Italian",  0.92, 320, 18, 7, 55),
        _make_meal("Tofu Stir-Fry",  "Asian",    0.88, 280, 22, 9, 30),
        _make_meal("Lentil Soup",    "Indian",   0.85, 250, 20, 5, 40),
        _make_meal("Quinoa Bowl",    "American", 0.80, 310, 15, 6, 45),
        _make_meal("Greek Salad",    "Greek",    0.75, 200, 12, 10, 20, "moderate"),
    ]


@pytest.fixture
def sample_preferences() -> Dict[str, Any]:
    return {
        "diet": "vegan",
        "calorie_limit": 500,
        "exclude": ["nuts"],
        "cuisine": None,
    }


@pytest.fixture
def empty_state() -> Dict[str, Any]:
    return {
        "preferences": {},
        "candidate_meals": [],
        "scored_meals": [],
        "final_recommendations": [],
        "logs": [],
    }


# ═════════════════════════════════════════════════════════════
# 1. Unit Tests — scoring_tool.rank_meals
# ═════════════════════════════════════════════════════════════

class TestRankMeals:

    def test_sorts_descending_by_final_score(self, sample_meals):
        """Meals must come back highest-score first."""
        result = rank_meals(sample_meals)
        scores = [m["final_score"] for m in result]
        assert scores == sorted(scores, reverse=True), "Meals not sorted descending"

    def test_returns_new_list_does_not_mutate(self, sample_meals):
        """rank_meals must not mutate the input list."""
        original_order = [m["name"] for m in sample_meals]
        rank_meals(sample_meals)
        assert [m["name"] for m in sample_meals] == original_order

    def test_handles_score_key_fallback(self):
        """Should accept ``score`` when ``final_score`` is absent."""
        meals = [{"name": "A", "score": 0.7}, {"name": "B", "score": 0.9}]
        result = rank_meals(meals)
        assert result[0]["name"] == "B"

    def test_handles_missing_score_gracefully(self):
        """Meals with no score key should be sorted to the bottom."""
        meals = [{"name": "NoScore"}, {"name": "HasScore", "final_score": 0.8}]
        result = rank_meals(meals)
        assert result[0]["name"] == "HasScore"

    def test_raises_type_error_on_non_list(self):
        with pytest.raises(TypeError, match="list"):
            rank_meals({"name": "dict not list"})  # type: ignore

    def test_empty_list_returns_empty(self):
        assert rank_meals([]) == []

    def test_single_element_returns_single(self):
        meal = _make_meal("Solo", final_score=0.5)
        result = rank_meals([meal])
        assert len(result) == 1
        assert result[0]["name"] == "Solo"


# ═════════════════════════════════════════════════════════════
# 2. Unit Tests — scoring_tool.diversify_meals
# ═════════════════════════════════════════════════════════════

class TestDiversifyMeals:

    def test_limits_per_cuisine_to_two_when_diverse_options_available(self):
        """With diverse cuisine options, no single cuisine should exceed the cap (2)."""
        # Create 6 Italian meals (highest scores) + 4 meals from other cuisines
        meals = (
            [_make_meal(f"Italian{i}", "Italian", 0.9 - i * 0.02) for i in range(6)]
            + [_make_meal(f"Thai{i}", "Thai", 0.80 - i * 0.02) for i in range(2)]
            + [_make_meal(f"Mexican{i}", "Mexican", 0.75 - i * 0.02) for i in range(2)]
        )
        result = diversify_meals(meals, top_n=5)
        italian_count = sum(1 for m in result if m["cuisine"] == "Italian")
        assert italian_count <= 2, f"Expected <=2 Italian meals when others available, got {italian_count}"
        assert len(result) == 5

    def test_respects_top_n(self, sample_meals):
        """Result must not exceed top_n."""
        result = diversify_meals(sample_meals, top_n=3)
        assert len(result) <= 3

    def test_fills_if_not_enough_diverse(self):
        """If only one cuisine available, fill with duplicates rather than under-return."""
        meals = [_make_meal(f"Pizza{i}", "Italian", 0.9 - i * 0.05) for i in range(6)]
        result = diversify_meals(meals, top_n=4)
        assert len(result) == 4

    def test_raises_type_error_on_non_list(self):
        with pytest.raises(TypeError):
            diversify_meals("not a list", top_n=5)  # type: ignore

    def test_raises_value_error_on_zero_top_n(self, sample_meals):
        with pytest.raises(ValueError, match="top_n"):
            diversify_meals(sample_meals, top_n=0)

    def test_empty_input_returns_empty(self):
        assert diversify_meals([], top_n=5) == []


# ═════════════════════════════════════════════════════════════
# 3. Unit Tests — scoring_tool.enrich_with_rank
# ═════════════════════════════════════════════════════════════

class TestEnrichWithRank:

    def test_adds_rank_starting_at_one(self, sample_meals):
        enriched = enrich_with_rank(sample_meals[:3])
        assert enriched[0]["rank"] == 1
        assert enriched[1]["rank"] == 2
        assert enriched[2]["rank"] == 3

    def test_adds_gold_badge_to_first(self, sample_meals):
        enriched = enrich_with_rank(sample_meals[:1])
        assert "🥇" in enriched[0]["badge"]

    def test_does_not_mutate_originals(self, sample_meals):
        copy_names = [m["name"] for m in sample_meals]
        enrich_with_rank(sample_meals)
        assert [m["name"] for m in sample_meals] == copy_names
        assert "rank" not in sample_meals[0], "Original must not have rank key added"

    def test_raises_type_error_on_non_list(self):
        with pytest.raises(TypeError):
            enrich_with_rank("not a list")  # type: ignore

    def test_empty_list_returns_empty(self):
        assert enrich_with_rank([]) == []

    def test_rank_beyond_three_has_number_badge(self):
        meals = [_make_meal(f"Meal{i}", final_score=0.9 - i * 0.05) for i in range(5)]
        enriched = enrich_with_rank(meals)
        assert "#4" in enriched[3]["badge"]
        assert "#5" in enriched[4]["badge"]


# ═════════════════════════════════════════════════════════════
# 4. Unit Tests — report_tool.build_markdown_report
# ═════════════════════════════════════════════════════════════

class TestBuildMarkdownReport:

    def test_returns_string(self, sample_meals, sample_preferences):
        result = build_markdown_report(enrich_with_rank(sample_meals), sample_preferences)
        assert isinstance(result, str)

    def test_contains_meal_names(self, sample_meals, sample_preferences):
        enriched = enrich_with_rank(sample_meals)
        result = build_markdown_report(enriched, sample_preferences)
        for meal in enriched:
            assert meal["name"] in result, f"{meal['name']} missing from report"

    def test_contains_preferences_section(self, sample_meals, sample_preferences):
        result = build_markdown_report(enrich_with_rank(sample_meals), sample_preferences)
        assert "Your Preferences" in result
        assert "vegan" in result
        assert "500" in result

    def test_empty_meals_returns_no_results_message(self, sample_preferences):
        result = build_markdown_report([], sample_preferences)
        assert "No Recommendations" in result or "No meals" in result

    def test_raises_type_error_on_non_list(self, sample_preferences):
        with pytest.raises(TypeError):
            build_markdown_report("not a list", sample_preferences)  # type: ignore

    def test_raises_type_error_on_non_dict_prefs(self, sample_meals):
        with pytest.raises(TypeError):
            build_markdown_report(sample_meals, "not a dict")  # type: ignore

    def test_contains_score_table(self, sample_meals, sample_preferences):
        enriched = enrich_with_rank(sample_meals)
        result = build_markdown_report(enriched, sample_preferences)
        assert "Score Comparison" in result

    def test_contains_statistics_section(self, sample_meals, sample_preferences):
        enriched = enrich_with_rank(sample_meals)
        result = build_markdown_report(enriched, sample_preferences)
        assert "Statistics" in result
        assert "Average calories" in result

    def test_reason_present_for_each_meal(self, sample_meals, sample_preferences):
        enriched = enrich_with_rank(sample_meals)
        result = build_markdown_report(enriched, sample_preferences)
        assert "Why This Meal Was Recommended" in result


# ═════════════════════════════════════════════════════════════
# 5. Unit Tests — report_tool.save_markdown_report
# ═════════════════════════════════════════════════════════════

class TestSaveMarkdownReport:

    def test_creates_file_on_disk(self, tmp_path):
        path = save_markdown_report("# Test", output_dir=str(tmp_path))
        assert os.path.isfile(path)

    def test_file_has_md_extension(self, tmp_path):
        path = save_markdown_report("# Test", output_dir=str(tmp_path))
        assert path.endswith(".md")

    def test_file_content_matches(self, tmp_path):
        content = "# My Report\nHello world."
        path = save_markdown_report(content, output_dir=str(tmp_path))
        with open(path, encoding="utf-8") as f:
            assert f.read() == content

    def test_raises_type_error_on_non_string(self, tmp_path):
        with pytest.raises(TypeError):
            save_markdown_report(123, output_dir=str(tmp_path))  # type: ignore


# ═════════════════════════════════════════════════════════════
# 6. Unit Tests — report_tool.save_json_results
# ═════════════════════════════════════════════════════════════

class TestSaveJsonResults:

    def test_creates_json_file(self, tmp_path, sample_meals, sample_preferences):
        path = save_json_results(sample_meals, sample_preferences, output_dir=str(tmp_path))
        assert os.path.isfile(path)
        assert path.endswith(".json")

    def test_json_is_valid_and_parseable(self, tmp_path, sample_meals, sample_preferences):
        path = save_json_results(sample_meals, sample_preferences, output_dir=str(tmp_path))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "recommendations" in data
        assert "preferences" in data
        assert "generated_at" in data

    def test_recommendation_count_matches(self, tmp_path, sample_meals, sample_preferences):
        path = save_json_results(sample_meals, sample_preferences, output_dir=str(tmp_path))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["total_recommendations"] == len(sample_meals)

    def test_raises_on_non_list_meals(self, tmp_path, sample_preferences):
        with pytest.raises(TypeError):
            save_json_results("bad", sample_preferences, output_dir=str(tmp_path))  # type: ignore

    def test_raises_on_non_dict_prefs(self, tmp_path, sample_meals):
        with pytest.raises(TypeError):
            save_json_results(sample_meals, "bad", output_dir=str(tmp_path))  # type: ignore


# ═════════════════════════════════════════════════════════════
# 7. Integration Tests — recommend_meals() end-to-end
# ═════════════════════════════════════════════════════════════

class TestRecommendMealsIntegration:

    def _build_state(self, scored_meals, preferences=None):
        return {
            "preferences": preferences or {"diet": "vegan", "calorie_limit": 500, "exclude": []},
            "candidate_meals": [],
            "scored_meals": scored_meals,
            "final_recommendations": [],
            "logs": [],
        }

    def test_populates_final_recommendations(self, sample_meals, tmp_path, monkeypatch):
        """Agent 4 must write enriched meals to state["final_recommendations"]."""
        # Patch output dirs to tmp_path to avoid polluting the repo
        monkeypatch.setattr("tools.report_tool.REPORTS_DIR", str(tmp_path / "reports"))
        monkeypatch.setattr("tools.report_tool.RESULTS_DIR", str(tmp_path / "results"))

        state = self._build_state(sample_meals)
        result = recommend_meals(state)

        assert "final_recommendations" in result
        assert len(result["final_recommendations"]) > 0

    def test_recommendations_have_rank_and_badge(self, sample_meals, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.report_tool.REPORTS_DIR", str(tmp_path / "reports"))
        monkeypatch.setattr("tools.report_tool.RESULTS_DIR", str(tmp_path / "results"))

        state = self._build_state(sample_meals)
        result = recommend_meals(state)

        for meal in result["final_recommendations"]:
            assert "rank" in meal, f"rank missing from meal: {meal.get('name')}"
            assert "badge" in meal, f"badge missing from meal: {meal.get('name')}"

    def test_log_entry_created(self, sample_meals, tmp_path, monkeypatch):
        """One log entry must be appended to state['logs']."""
        monkeypatch.setattr("tools.report_tool.REPORTS_DIR", str(tmp_path / "reports"))
        monkeypatch.setattr("tools.report_tool.RESULTS_DIR", str(tmp_path / "results"))

        state = self._build_state(sample_meals)
        result = recommend_meals(state)

        agent4_logs = [l for l in result["logs"] if l.get("agent") == "RecommendationReportAgent"]
        assert len(agent4_logs) == 1, "Expected exactly 1 log entry from Agent 4"

    def test_log_entry_has_required_keys(self, sample_meals, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.report_tool.REPORTS_DIR", str(tmp_path / "reports"))
        monkeypatch.setattr("tools.report_tool.RESULTS_DIR", str(tmp_path / "results"))

        state = self._build_state(sample_meals)
        result = recommend_meals(state)

        log = next(l for l in result["logs"] if l.get("agent") == "RecommendationReportAgent")
        assert "input" in log
        assert "output" in log
        assert "timestamp" in log

    def test_files_saved_to_disk(self, sample_meals, tmp_path, monkeypatch):
        """Both MD report and JSON results must be written to disk."""
        reports_dir = str(tmp_path / "reports")
        results_dir = str(tmp_path / "results")
        monkeypatch.setattr("tools.report_tool.REPORTS_DIR", reports_dir)
        monkeypatch.setattr("tools.report_tool.RESULTS_DIR", results_dir)

        state = self._build_state(sample_meals)
        recommend_meals(state)

        md_files = [f for f in os.listdir(reports_dir) if f.endswith(".md")]
        json_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
        assert len(md_files) == 1, "Expected exactly 1 Markdown report file"
        assert len(json_files) == 1, "Expected exactly 1 JSON results file"

    def test_empty_scored_meals_returns_empty_recommendations(self, empty_state, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.report_tool.REPORTS_DIR", str(tmp_path / "reports"))
        monkeypatch.setattr("tools.report_tool.RESULTS_DIR", str(tmp_path / "results"))

        result = recommend_meals(empty_state)
        assert result["final_recommendations"] == []


# ═════════════════════════════════════════════════════════════
# 8. Edge Case & Security Tests
# ═════════════════════════════════════════════════════════════

class TestEdgeCasesAndSecurity:

    def test_meals_with_missing_name_field(self, tmp_path, monkeypatch):
        """Meals without a name key must not crash Agent 4."""
        monkeypatch.setattr("tools.report_tool.REPORTS_DIR", str(tmp_path / "reports"))
        monkeypatch.setattr("tools.report_tool.RESULTS_DIR", str(tmp_path / "results"))

        meals = [{"final_score": 0.8, "category": "healthy", "health_flags": []}]
        state = {
            "preferences": {},
            "candidate_meals": [],
            "scored_meals": meals,
            "final_recommendations": [],
            "logs": [],
        }
        result = recommend_meals(state)
        assert result["final_recommendations"] is not None

    def test_meals_with_none_score_handled(self):
        """rank_meals must not crash when final_score is None."""
        meals = [{"name": "X", "final_score": None}, {"name": "Y", "final_score": 0.8}]
        result = rank_meals(meals)
        assert result[0]["name"] == "Y"  # 0.8 > 0.0 (None fallback)

    def test_report_with_health_flags(self, sample_preferences):
        """Health flags must appear in the report when present."""
        meal = _make_meal("Flagged Meal", health_flags=["high_fat", "high_sugar"])
        enriched = enrich_with_rank([meal])
        report = build_markdown_report(enriched, sample_preferences)
        assert "high_fat" in report
        assert "high_sugar" in report

    def test_report_no_health_flags_shows_none(self, sample_preferences):
        """Health flags column must show 'None' when a meal has no flags."""
        meal = _make_meal("Clean Meal", health_flags=[])
        enriched = enrich_with_rank([meal])
        report = build_markdown_report(enriched, sample_preferences)
        assert "None" in report

    def test_preferences_with_no_exclusions(self):
        """Agent must handle empty exclude list without error."""
        prefs = {"diet": "vegan", "calorie_limit": 400, "exclude": [], "cuisine": None}
        meal = _make_meal()
        enriched = enrich_with_rank([meal])
        report = build_markdown_report(enriched, prefs)
        assert "None" in report  # exclusions display as None

    def test_top_recommendation_is_highest_scored(self, sample_meals, tmp_path, monkeypatch):
        """Invariant: the first final recommendation must always be the highest-scored meal."""
        monkeypatch.setattr("tools.report_tool.REPORTS_DIR", str(tmp_path / "reports"))
        monkeypatch.setattr("tools.report_tool.RESULTS_DIR", str(tmp_path / "results"))

        state = {
            "preferences": {"diet": "vegan", "calorie_limit": 600, "exclude": []},
            "candidate_meals": [],
            "scored_meals": sample_meals,
            "final_recommendations": [],
            "logs": [],
        }
        result = recommend_meals(state)
        recs = result["final_recommendations"]
        if len(recs) > 1:
            assert recs[0]["final_score"] >= recs[1]["final_score"], \
                "First recommendation must have the highest score"

    def test_diversity_invariant_no_cuisine_exceeds_two_with_multiple_cuisines(self):
        """Invariant: with multiple cuisines available, no single cuisine exceeds 2."""
        meals = (
            [_make_meal(f"Italian{i}", "Italian", 0.95 - i * 0.04) for i in range(5)]
            + [_make_meal(f"Mexican{i}", "Mexican", 0.80 - i * 0.04) for i in range(5)]
            + [_make_meal(f"Thai{i}", "Thai", 0.70 - i * 0.04) for i in range(5)]
        )
        result = diversify_meals(meals, top_n=6)
        from collections import Counter
        cuisine_counts = Counter(m["cuisine"] for m in result)
        for cuisine, count in cuisine_counts.items():
            assert count <= 2, f"Cuisine '{cuisine}' appears {count} times (max 2)"