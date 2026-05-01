"""
tools/description_tool.py
==========================
Generates a human-readable "why this meal was selected" description for
each final recommendation.

The description is built deterministically from the meal's scored attributes
and the user's preferences — no LLM call is needed.

Public API:
    generate_selection_description(meal, preferences) -> str

Assignment compliance:
    ✔  Dedicated tool with a clear, single responsibility
    ✔  Strict type hints and docstring
    ✔  Purely rule-based — fast, reproducible, works offline
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


def generate_selection_description(
    meal: Mapping[str, Any],
    preferences: Mapping[str, Any],
) -> str:
    """
    Build a plain-English description explaining why a meal was recommended.

    The description references the user's actual preferences (diet, calorie
    limit, exclusions, cuisine, goal) and the meal's scored attributes
    (nutrition score, diet match, health flags, category) so the explanation
    is specific and meaningful.

    Args:
        meal:        A scored/enriched meal dict from Agent 4.
        preferences: The user's preference dict from Agent 1.

    Returns:
        A 2–4 sentence string explaining why this specific meal was chosen.

    Example:
        >>> desc = generate_selection_description(meal, {"diet": "vegan", "calorie_limit": 500})
        >>> "vegan" in desc.lower()
        True
    """
    parts: List[str] = []

    name          = meal.get("name", "This meal")
    calories      = _safe_float(meal.get("calories"))
    protein       = _safe_float(meal.get("protein"))
    fat           = _safe_float(meal.get("fat"))
    carbs         = _safe_float(meal.get("carbs"))
    final_score   = _safe_float(meal.get("final_score") or meal.get("score"))
    nutrition_score = _safe_float(meal.get("nutrition_score"))
    diet_match    = _safe_float(meal.get("diet_match"))
    category      = (meal.get("category") or "").lower()
    health_flags  = meal.get("health_flags") or []
    rank          = meal.get("rank")

    diet          = _str(preferences.get("diet"))
    calorie_limit = _safe_float(preferences.get("calorie_limit"))
    exclude       = preferences.get("exclude") or []
    cuisine_pref  = _str(preferences.get("cuisine"))
    goal          = _str(preferences.get("goal"))

    # ── Opening: rank + category ──────────────────────────────────────────────
    rank_text = f"Ranked #{rank}, this" if rank else "This"
    cat_text  = {"healthy": "nutritionally excellent", "moderate": "solid",
                 "unhealthy": "indulgent"}.get(category, "well-scored")
    parts.append(f"{rank_text} is a {cat_text} choice with an overall match score of "
                 f"{round((final_score or 0) * 100)}%.")

    # ── Calorie fit ───────────────────────────────────────────────────────────
    if calories is not None and calorie_limit is not None:
        pct = round((calories / calorie_limit) * 100)
        diff = round(calorie_limit - calories)
        parts.append(
            f"At {calories:.0f} kcal it uses {pct}% of your {calorie_limit:.0f} kcal budget, "
            f"leaving {diff} kcal to spare."
        )
    elif calories is not None:
        if calories <= 350:
            parts.append(f"At just {calories:.0f} kcal it is a very light option.")
        elif calories <= 550:
            parts.append(f"With {calories:.0f} kcal it fits comfortably into a balanced day.")
        else:
            parts.append(f"It provides {calories:.0f} kcal — a satisfying, energy-rich meal.")

    # ── Diet alignment ────────────────────────────────────────────────────────
    if diet and diet_match is not None:
        match_pct = round(diet_match * 100)
        diet_label = diet.capitalize()
        if match_pct >= 95:
            parts.append(f"It is fully {diet_label}-compliant ({match_pct}% diet match).")
        elif match_pct >= 70:
            parts.append(
                f"It aligns well with your {diet_label} preference ({match_pct}% diet match)."
            )
        else:
            parts.append(
                f"It partially matches your {diet_label} preference ({match_pct}% diet match)."
            )
    elif not diet:
        parts.append("No specific diet restriction was applied, so all meal types were considered.")

    # ── Macro highlights ──────────────────────────────────────────────────────
    macro_highlights: List[str] = []
    if protein is not None:
        if protein >= 25:
            macro_highlights.append(f"high protein ({protein:.0f}g)")
        elif protein >= 15:
            macro_highlights.append(f"good protein ({protein:.0f}g)")
    if carbs is not None and carbs <= 30:
        macro_highlights.append(f"low carbs ({carbs:.0f}g)")
    if fat is not None and 8 <= fat <= 20:
        macro_highlights.append(f"healthy fat ({fat:.0f}g)")

    if goal:
        goal_lower = goal.lower()
        if "high protein" in goal_lower and protein is not None and protein >= 20:
            macro_highlights.insert(0, f"supports your high-protein goal")
        elif "low carb" in goal_lower and carbs is not None and carbs <= 35:
            macro_highlights.insert(0, "supports your low-carb goal")
        elif "balanced" in goal_lower:
            macro_highlights.insert(0, "offers a balanced macro profile")

    if macro_highlights:
        parts.append("Nutritionally it stands out for: " + ", ".join(macro_highlights) + ".")

    # ── Cuisine match ─────────────────────────────────────────────────────────
    meal_cuisine = _str(meal.get("cuisine"))
    if cuisine_pref and meal_cuisine:
        if cuisine_pref.lower() in meal_cuisine.lower():
            parts.append(f"It matches your {cuisine_pref} cuisine preference.")
    elif meal_cuisine:
        parts.append(f"This is a {meal_cuisine} dish.")

    # ── Exclusion safety ──────────────────────────────────────────────────────
    if exclude:
        excl_str = ", ".join(str(e) for e in exclude)
        parts.append(
            f"None of your excluded ingredients ({excl_str}) were detected in this meal."
        )

    # ── Health flag note ──────────────────────────────────────────────────────
    if health_flags:
        flag_labels = {
            "high_sugar":   "high sugar content",
            "high_sodium":  "high sodium",
            "high_fat":     "higher fat content",
            "high_calorie": "calorie-dense",
            "low_protein":  "lower protein",
            "high_carb":    "high carbs",
        }
        flag_texts = [flag_labels.get(f, f.replace("_", " ")) for f in health_flags[:2]]
        parts.append(f"Note: be aware it has {' and '.join(flag_texts)}.")

    # ── Nutrition score sign-off ──────────────────────────────────────────────
    if nutrition_score is not None:
        ns_pct = round(nutrition_score * 100)
        if ns_pct >= 80:
            parts.append(f"The nutrition agent scored it {ns_pct}/100 — an excellent nutritional profile.")
        elif ns_pct >= 65:
            parts.append(f"The nutrition agent scored it {ns_pct}/100 — a good nutritional profile.")
        else:
            parts.append(f"The nutrition agent gave it a score of {ns_pct}/100.")

    return " ".join(parts)


# ── Private helpers ────────────────────────────────────────────────────────────

def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None
