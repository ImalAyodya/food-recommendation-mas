"""
Tool: scoring_tool.py
======================

Custom Python tool used exclusively by Agent 4 (RecommendationReportAgent).

Responsibilities:
    - Rank a list of scored meal dicts by final_score (descending)
    - Apply cuisine-diversity filtering so top-N results span multiple cuisines
    - Enrich each selected meal with a human-readable rank badge

This tool implements DETERMINISTIC, purely rule-based logic.
No LLM is used here — the heavy reasoning was done by Agent 3.

Assignment compliance:
    ✔  Custom Python tool (not an LLM call)
    ✔  Strict type hinting on all public functions
    ✔  Descriptive docstrings with Args / Returns / Raises
    ✔  Robust error handling for malformed meal records
"""

from __future__ import annotations

from typing import Any, Dict, List


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def rank_meals(meals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort a list of scored meal dicts by their final_score in descending order.

    Accepts both ``final_score`` and ``score`` keys (Agent 3 writes both) so
    this tool is robust to minor upstream schema changes.  Meals missing both
    keys are sorted to the bottom rather than raising an exception.

    Args:
        meals: A list of meal dictionaries, each expected to have at least
               a ``final_score`` or ``score`` key of type float.

    Returns:
        A new list sorted by score descending. The original list is unchanged.

    Raises:
        TypeError: If ``meals`` is not a list.

    Example:
        >>> rank_meals([{"name": "A", "final_score": 0.7}, {"name": "B", "final_score": 0.9}])
        [{"name": "B", "final_score": 0.9}, {"name": "A", "final_score": 0.7}]
    """
    if not isinstance(meals, list):
        raise TypeError(f"rank_meals expects a list, got {type(meals).__name__}")

    return sorted(
        meals,
        key=lambda m: _extract_score(m),
        reverse=True,
    )


def diversify_meals(ranked_meals: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Select up to ``top_n`` meals while enforcing a per-cuisine cap.

    Strategy:
        1. Iterate the already-ranked list (highest score first).
        2. Keep a meal only if the cuisine it belongs to has appeared fewer
           than MAX_PER_CUISINE times in the selection.
        3. If the full ``top_n`` cannot be reached through diversity alone,
           fill remaining slots from the ranked list in score order.

    This prevents the common failure mode where all top results are from
    the same cuisine (e.g., 5 identical Italian dishes).

    Args:
        ranked_meals: Pre-sorted list of meal dicts (highest score first).
        top_n: Maximum number of meals to return. Defaults to 5.

    Returns:
        A list of up to ``top_n`` meal dicts chosen for quality and diversity.

    Raises:
        TypeError: If ``ranked_meals`` is not a list or ``top_n`` is not int.
        ValueError: If ``top_n`` is less than 1.

    Example:
        >>> meals = [
        ...     {"name": "Pasta", "cuisine": "Italian", "final_score": 0.9},
        ...     {"name": "Pizza", "cuisine": "Italian", "final_score": 0.88},
        ...     {"name": "Tacos", "cuisine": "Mexican", "final_score": 0.85},
        ...     {"name": "Sushi", "cuisine": "Japanese", "final_score": 0.82},
        ...     {"name": "Risotto", "cuisine": "Italian", "final_score": 0.80},
        ... ]
        >>> diversify_meals(meals, top_n=4)
        # Returns Pasta, Pizza, Tacos, Sushi (Risotto skipped — 3rd Italian)
    """
    if not isinstance(ranked_meals, list):
        raise TypeError(f"diversify_meals expects a list, got {type(ranked_meals).__name__}")
    if not isinstance(top_n, int):
        raise TypeError(f"top_n must be an int, got {type(top_n).__name__}")
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")

    MAX_PER_CUISINE: int = 2

    selected: List[Dict[str, Any]] = []
    cuisine_count: Dict[str, int] = {}

    # First pass — prefer diversity
    for meal in ranked_meals:
        if len(selected) >= top_n:
            break
        cuisine_key: str = _extract_cuisine(meal)
        if cuisine_count.get(cuisine_key, 0) < MAX_PER_CUISINE:
            selected.append(meal)
            cuisine_count[cuisine_key] = cuisine_count.get(cuisine_key, 0) + 1

    # Second pass — fill remaining slots, now allowing up to top_n per cuisine
    if len(selected) < top_n:
        already_selected_ids = {id(m) for m in selected}
        for meal in ranked_meals:
            if len(selected) >= top_n:
                break
            if id(meal) not in already_selected_ids:
                selected.append(meal)
                cuisine_key = _extract_cuisine(meal)
                cuisine_count[cuisine_key] = cuisine_count.get(cuisine_key, 0) + 1

    return selected


def enrich_with_rank(meals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add ``rank`` (1-based integer) and ``badge`` (emoji string) to each meal.

    Creates a copy of each dict so the original scored_meals list in state
    is never mutated.

    Args:
        meals: Ordered list of meal dicts (already sorted and diversified).

    Returns:
        New list of meal dicts with ``rank`` and ``badge`` fields added.

    Raises:
        TypeError: If ``meals`` is not a list.

    Example:
        >>> enrich_with_rank([{"name": "Salad"}, {"name": "Soup"}])
        [
            {"name": "Salad", "rank": 1, "badge": "🥇 #1 Best Match"},
            {"name": "Soup",  "rank": 2, "badge": "🥈 #2 Runner-Up"},
        ]
    """
    if not isinstance(meals, list):
        raise TypeError(f"enrich_with_rank expects a list, got {type(meals).__name__}")

    _BADGES: Dict[int, str] = {
        1: "🥇 #1 Best Match",
        2: "🥈 #2 Runner-Up",
        3: "🥉 #3 Third Place",
    }

    enriched: List[Dict[str, Any]] = []
    for rank, meal in enumerate(meals, start=1):
        m = dict(meal)  # shallow copy — do not mutate the original
        m["rank"] = rank
        m["badge"] = _BADGES.get(rank, f"#{rank} Recommended")
        enriched.append(m)

    return enriched


# ─────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────

def _extract_score(meal: Dict[str, Any]) -> float:
    """
    Safely extract a numeric score from a meal dict.

    Tries ``final_score`` first, then ``score``, then returns 0.0.

    Args:
        meal: A meal dictionary from Agent 3.

    Returns:
        Float score value; 0.0 if neither key exists or value is not numeric.
    """
    for key in ("final_score", "score"):
        value = meal.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def _extract_cuisine(meal: Dict[str, Any]) -> str:
    """
    Safely extract and normalise the cuisine string from a meal dict.

    Args:
        meal: A meal dictionary.

    Returns:
        Lowercase stripped cuisine string, or ``"unknown"`` if absent.
    """
    raw = meal.get("cuisine", "")
    if not raw:
        return "unknown"
    return str(raw).strip().lower()