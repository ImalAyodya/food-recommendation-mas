from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple
import json
import re

from tools.logger import log_agent_step


_ANIMAL_TERMS = {
    "anchovy",
    "bacon",
    "beef",
    "chicken",
    "fish",
    "gelatin",
    "ham",
    "lamb",
    "lard",
    "meat",
    "pork",
    "prawn",
    "salmon",
    "shrimp",
    "tuna",
    "turkey",
}

_NON_VEGAN_TERMS = _ANIMAL_TERMS.union(
    {
        "butter",
        "casein",
        "cheese",
        "cream",
        "egg",
        "honey",
        "milk",
        "whey",
        "yogurt",
    }
)


def analyze_nutrition(state: Dict[str, Any]) -> Dict[str, Any]:
    preferences = state.get("preferences", {})
    if not isinstance(preferences, Mapping):
        preferences = {}

    candidate_meals = state.get("candidate_meals", [])
    if not isinstance(candidate_meals, list):
        candidate_meals = []

    evaluations: List[Dict[str, Any]] = []
    scored_valid_meals: List[Dict[str, Any]] = []

    for meal in candidate_meals:
        if not isinstance(meal, Mapping):
            evaluations.append(
                {
                    "meal": "Unknown Meal",
                    "rejected": True,
                    "rejection_reason": "Meal record format is invalid.",
                }
            )
            continue

        meal_dict = dict(meal)
        evaluation = _evaluate_meal(meal_dict, preferences)
        evaluations.append(evaluation)

        if not evaluation.get("rejected", True):
            scored_valid_meals.append(evaluation)

    state["nutrition_evaluations"] = evaluations
    state["scored_meals"] = scored_valid_meals

    log_agent_step(
        state=state,
        agent_name="NutritionAnalyzerAgent",
        input_data={
            "preferences": dict(preferences),
            "candidate_count": len(candidate_meals),
        },
        output_data={
            "evaluated_count": len(evaluations),
            "accepted_count": len(scored_valid_meals),
            "rejected_count": len(evaluations) - len(scored_valid_meals),
        },
    )

    print(_build_agent3_console_report(preferences=preferences, evaluations=evaluations))

    return state


def _build_agent3_console_report(
    preferences: Mapping[str, Any],
    evaluations: List[Dict[str, Any]],
) -> str:
    accepted_meals: List[Dict[str, Any]] = []
    rejected_count = 0

    for item in evaluations:
        if item.get("rejected", True):
            rejected_count += 1
            continue

        accepted_meals.append(
            {
                "meal": item.get("meal", "Unknown Meal"),
                "nutrition_score": item.get("nutrition_score"),
                "diet_match": item.get("diet_match"),
                "final_score": item.get("final_score"),
                "category": item.get("category"),
                "health_flags": item.get("health_flags", []),
                "reason": item.get("reason", ""),
            }
        )

    report = {
        "preferences": dict(preferences),
        "summary": {
            "evaluated": len(evaluations),
            "accepted": len(accepted_meals),
            "rejected": rejected_count,
        },
        "accepted_meals": accepted_meals,
    }
    report_json = json.dumps(
        report,
        indent=2,
        ensure_ascii=True,
        default=str,
    )
    footer = (
        "Agent 3: Totals -> "
        f"Accepted={len(accepted_meals)}, Rejected={rejected_count}"
    )

    return "Agent 3: Nutrition Analyzer Report\n" + report_json + "\n" + footer


def _evaluate_meal(meal: Dict[str, Any], preferences: Mapping[str, Any]) -> Dict[str, Any]:
    meal_name = _normalize_text(meal.get("name")) or "Unknown Meal"
    calorie_limit = _to_float(preferences.get("calorie_limit"))
    diet_preference = _normalize_text(preferences.get("diet"))
    goal = _normalize_text(preferences.get("goal"))
    exclude_terms = _normalize_list(preferences.get("exclude"))

    calories = _to_float(meal.get("calories"))
    protein = _to_float(meal.get("protein"))
    carbs = _to_float(meal.get("carbs"))
    fat = _to_float(meal.get("fat"))
    sugar = _to_float(meal.get("sugar"))
    sodium = _to_float(meal.get("sodium"))

    critical_fields = {
        "calories": calories,
        "protein": protein,
        "carbs": carbs,
        "fat": fat,
    }
    missing_critical = [key for key, value in critical_fields.items() if value is None]
    if missing_critical:
        return {
            "meal": meal_name,
            "rejected": True,
            "rejection_reason": (
                "Missing critical nutrition data: " + ", ".join(sorted(missing_critical)) + "."
            ),
        }

    if calorie_limit is not None and calories > calorie_limit:
        return {
            "meal": meal_name,
            "rejected": True,
            "rejection_reason": "Exceeds the calorie limit.",
        }

    exclusion_hit = _find_exclusion_conflict(meal, exclude_terms)
    if exclusion_hit is not None:
        return {
            "meal": meal_name,
            "rejected": True,
            "rejection_reason": f"Contains excluded ingredient or allergen: {exclusion_hit}.",
        }

    diet_rejected, diet_reason, diet_match = _evaluate_diet_match(
        meal=meal,
        diet_preference=diet_preference,
        carbs=carbs,
    )
    if diet_rejected:
        return {
            "meal": meal_name,
            "rejected": True,
            "rejection_reason": diet_reason,
        }

    health_flags = _collect_health_flags(
        calories=calories,
        protein=protein,
        carbs=carbs,
        fat=fat,
        sugar=sugar,
        sodium=sodium,
        calorie_limit=calorie_limit,
    )

    nutrition_score = _compute_nutrition_score(
        calories=calories,
        protein=protein,
        carbs=carbs,
        fat=fat,
        calorie_limit=calorie_limit,
        goal=goal,
        diet_preference=diet_preference,
        health_flags=health_flags,
    )

    final_score = _round_2((0.5 * nutrition_score) + (0.5 * diet_match))
    category = _categorize_meal(final_score, nutrition_score, health_flags)
    reason = _build_reason(
        diet_preference=diet_preference,
        goal=goal,
        calories=calories,
        protein=protein,
        carbs=carbs,
        calorie_limit=calorie_limit,
        health_flags=health_flags,
    )

    evaluated_meal = dict(meal)
    evaluated_meal.update(
        {
            "meal": meal_name,
            "name": meal_name,
            "rejected": False,
            "nutrition_score": _round_2(nutrition_score),
            "diet_match": _round_2(diet_match),
            "health_flags": health_flags,
            "final_score": final_score,
            "score": final_score,
            "category": category,
            "reason": reason,
        }
    )
    return evaluated_meal


def _evaluate_diet_match(
    meal: Mapping[str, Any],
    diet_preference: Optional[str],
    carbs: float,
) -> Tuple[bool, str, float]:
    if not diet_preference:
        return False, "", 1.0

    diet_tokens = _normalize_tokens(meal.get("diet_type"))
    ingredient_text = _compose_meal_text(meal)

    if diet_preference == "vegan":
        if _contains_any_term(ingredient_text, _NON_VEGAN_TERMS):
            return True, "Conflicts with vegan dietary requirement.", 0.0
        if "vegan" in diet_tokens:
            return False, "", 1.0
        if "vegetarian" in diet_tokens:
            return False, "", 0.7
        return False, "", 0.75

    if diet_preference == "vegetarian":
        if _contains_any_term(ingredient_text, _ANIMAL_TERMS):
            return True, "Conflicts with vegetarian dietary requirement.", 0.0
        if "vegetarian" in diet_tokens or "vegan" in diet_tokens:
            return False, "", 1.0
        return False, "", 0.8

    if diet_preference == "keto":
        if carbs > 50:
            return True, "Conflicts with keto preference due to high carbs.", 0.0
        if carbs <= 20:
            return False, "", 1.0
        if carbs <= 35:
            return False, "", 0.85
        return False, "", 0.7

    if diet_preference in diet_tokens:
        return False, "", 1.0

    if diet_tokens:
        return True, f"Does not match required diet: {diet_preference}.", 0.0

    return False, "", 0.6


def _collect_health_flags(
    calories: float,
    protein: float,
    carbs: float,
    fat: float,
    sugar: Optional[float],
    sodium: Optional[float],
    calorie_limit: Optional[float],
) -> List[str]:
    flags: List[str] = []

    if sugar is not None and sugar > 25:
        flags.append("high_sugar")
    if sodium is not None and sodium > 700:
        flags.append("high_sodium")
    if fat > 30:
        flags.append("high_fat")
    if calorie_limit is not None:
        if calories > 0.9 * calorie_limit:
            flags.append("high_calorie")
    elif calories > 800:
        flags.append("high_calorie")
    if protein < 12:
        flags.append("low_protein")
    if carbs > 90:
        flags.append("high_carb")

    return flags


def _compute_nutrition_score(
    calories: float,
    protein: float,
    carbs: float,
    fat: float,
    calorie_limit: Optional[float],
    goal: Optional[str],
    diet_preference: Optional[str],
    health_flags: List[str],
) -> float:
    goal_text = goal or ""
    is_high_protein_goal = "high protein" in goal_text
    is_low_calorie_goal = "low calorie" in goal_text
    is_low_carb_goal = "low carb" in goal_text
    is_balanced_goal = "balanced" in goal_text

    if calorie_limit is not None and calorie_limit > 0:
        calorie_ratio = calories / calorie_limit
        if calorie_ratio <= 0.7:
            calorie_score = 1.0
        elif calorie_ratio <= 0.85:
            calorie_score = 0.9
        else:
            calorie_score = 0.75
    else:
        if calories <= 450:
            calorie_score = 1.0
        elif calories <= 650:
            calorie_score = 0.85
        elif calories <= 850:
            calorie_score = 0.65
        else:
            calorie_score = 0.45

    if is_high_protein_goal:
        if protein >= 30:
            protein_score = 1.0
        elif protein >= 22:
            protein_score = 0.85
        elif protein >= 15:
            protein_score = 0.65
        else:
            protein_score = 0.4
    else:
        if protein >= 25:
            protein_score = 1.0
        elif protein >= 18:
            protein_score = 0.85
        elif protein >= 12:
            protein_score = 0.7
        else:
            protein_score = 0.5

    if is_low_carb_goal or diet_preference == "keto":
        if carbs <= 20:
            carb_score = 1.0
        elif carbs <= 35:
            carb_score = 0.8
        elif carbs <= 50:
            carb_score = 0.6
        else:
            carb_score = 0.35
    elif is_balanced_goal:
        if 25 <= carbs <= 55:
            carb_score = 0.95
        elif 15 <= carbs <= 70:
            carb_score = 0.75
        else:
            carb_score = 0.55
    else:
        if carbs <= 30:
            carb_score = 0.9
        elif carbs <= 60:
            carb_score = 0.85
        elif carbs <= 80:
            carb_score = 0.7
        else:
            carb_score = 0.5

    if 10 <= fat <= 25:
        fat_score = 0.9
    elif fat < 10:
        fat_score = 0.75
    elif fat <= 35:
        fat_score = 0.65
    else:
        fat_score = 0.45

    base_score = (
        (0.30 * calorie_score)
        + (0.30 * protein_score)
        + (0.20 * carb_score)
        + (0.20 * fat_score)
    )

    if is_low_calorie_goal and calorie_limit is not None and calories <= 0.75 * calorie_limit:
        base_score += 0.05
    if is_high_protein_goal and protein >= 25:
        base_score += 0.05
    if is_low_carb_goal and carbs <= 25:
        base_score += 0.05
    if is_balanced_goal and (15 <= protein <= 35) and (20 <= carbs <= 60) and (10 <= fat <= 25):
        base_score += 0.05

    score_after_penalty = base_score - (0.05 * len(health_flags))
    return _clip(score_after_penalty, 0.0, 1.0)


def _categorize_meal(final_score: float, nutrition_score: float, health_flags: List[str]) -> str:
    if final_score >= 0.75 and nutrition_score >= 0.7 and len(health_flags) <= 1:
        return "healthy"
    if final_score < 0.45 or nutrition_score < 0.45 or len(health_flags) >= 3:
        return "unhealthy"
    return "moderate"


def _build_reason(
    diet_preference: Optional[str],
    goal: Optional[str],
    calories: float,
    protein: float,
    carbs: float,
    calorie_limit: Optional[float],
    health_flags: List[str],
) -> str:
    positive_parts: List[str] = []
    goal_text = goal or ""

    if "high protein" in goal_text and protein >= 20:
        positive_parts.append("high protein")
    elif "low carb" in goal_text and carbs <= 35:
        positive_parts.append("low carb")
    elif "balanced" in goal_text and (15 <= protein <= 35) and (20 <= carbs <= 60):
        positive_parts.append("balanced macros")

    if calorie_limit is not None and calories <= calorie_limit:
        positive_parts.append("within calorie limit")

    if diet_preference:
        positive_parts.append(f"matches {diet_preference} preference")

    if not positive_parts:
        positive_parts.append("nutritionally suitable")

    sentence = _capitalize_first(" and ".join(positive_parts)) + "."

    if health_flags:
        flag_text = {
            "high_sugar": "high sugar",
            "high_sodium": "high sodium",
            "high_fat": "high fat",
            "high_calorie": "high calories",
            "low_protein": "low protein",
            "high_carb": "high carbs",
        }.get(health_flags[0], health_flags[0].replace("_", " "))
        sentence = sentence[:-1] + f" but has slightly {flag_text}."

    return sentence


def _find_exclusion_conflict(meal: Mapping[str, Any], exclude_terms: List[str]) -> Optional[str]:
    if not exclude_terms:
        return None

    meal_text = _compose_meal_text(meal)
    for term in exclude_terms:
        if _contains_term(meal_text, term):
            return term
    return None


def _compose_meal_text(meal: Mapping[str, Any]) -> str:
    chunks = [
        _normalize_text(meal.get("name")) or "",
        _normalize_text(meal.get("ingredients")) or "",
        _normalize_text(meal.get("allergens")) or "",
        _normalize_text(meal.get("diet_type")) or "",
    ]
    return " ".join(chunks).strip()


def _normalize_tokens(value: Any) -> List[str]:
    if value is None:
        return []
    text = _normalize_text(value)
    if not text:
        return []
    return [token for token in re.split(r"[\|,;/]+", text) if token]


def _normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [token for token in (_normalize_text(item) for item in value) if token]
    normalized = _normalize_text(value)
    return [normalized] if normalized else []


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _contains_any_term(text: str, terms: set[str]) -> bool:
    return any(_contains_term(text, term) for term in terms)


def _contains_term(text: str, term: str) -> bool:
    if not text or not term:
        return False
    pattern = rf"\b{re.escape(term)}\b"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _round_2(value: float) -> float:
    return round(value + 1e-10, 2)


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _capitalize_first(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]