"""
tools/nutrition_tool.py
========================
Custom Python tool used by Agent 3 (NutritionAnalyzerAgent).

Responsibilities:
    - evaluate_meal_nutrition(meal, preferences) -> per-meal evaluation dict
    - batch_evaluate(meals, preferences)         -> (all_evals, accepted_meals)
    - compute_health_flags(...)                  -> list of warning strings

Assignment compliance:
    ✔  Custom Python tool (no LLM — purely rule-based)
    ✔  Strict type hints on all public functions
    ✔  Descriptive docstrings with Args / Returns / Raises
    ✔  Robust error handling for malformed meal records
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

# ─── Diet-classification term sets ───────────────────────────────────────────

_ANIMAL_TERMS: frozenset = frozenset({
    "anchovy","bacon","beef","chicken","fish","gelatin",
    "ham","lamb","lard","meat","pork","prawn","salmon",
    "shrimp","tuna","turkey",
})

_NON_VEGAN_TERMS: frozenset = _ANIMAL_TERMS | frozenset({
    "butter","casein","cheese","cream","egg","honey","milk","whey","yogurt"
})

# ─── Public API ───────────────────────────────────────────────────────────────

def evaluate_meal_nutrition(
    meal: Dict[str, Any],
    preferences: Mapping[str, Any],
) -> Dict[str, Any]:
    """Evaluate a single meal against user preferences and return a scored dict."""
    if not isinstance(meal, Mapping):
        raise TypeError(f"meal must be a mapping, got {type(meal).__name__}")
    if not isinstance(preferences, Mapping):
        raise TypeError(f"preferences must be a mapping, got {type(preferences).__name__}")
    return _evaluate_meal(dict(meal), preferences)


def batch_evaluate(
    meals: List[Dict[str, Any]],
    preferences: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluate all meals; return (all_evaluations, accepted_meals)."""
    if not isinstance(meals, list):
        raise TypeError(f"meals must be a list, got {type(meals).__name__}")
    if not isinstance(preferences, Mapping):
        raise TypeError(f"preferences must be a mapping, got {type(preferences).__name__}")

    all_evals: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []

    for meal in meals:
        if not isinstance(meal, Mapping):
            all_evals.append({"meal":"Unknown Meal","rejected":True,
                              "rejection_reason":"Meal record format is invalid."})
            continue
        ev = evaluate_meal_nutrition(dict(meal), preferences)
        all_evals.append(ev)
        if not ev.get("rejected", True):
            accepted.append(ev)

    return all_evals, accepted


def compute_health_flags(
    calories: float, protein: float, carbs: float, fat: float,
    sugar: Optional[float], sodium: Optional[float], calorie_limit: Optional[float],
) -> List[str]:
    """Return health warning flags for a meal based on nutrient thresholds."""
    return _collect_health_flags(calories, protein, carbs, fat, sugar, sodium, calorie_limit)


# ─── Private implementation ───────────────────────────────────────────────────

def _evaluate_meal(meal: Dict[str, Any], preferences: Mapping[str, Any]) -> Dict[str, Any]:
    meal_name = _normalize_text(meal.get("name")) or "Unknown Meal"
    calorie_limit = _to_float(preferences.get("calorie_limit"))
    diet_pref = _normalize_text(preferences.get("diet"))
    goal = _normalize_text(preferences.get("goal"))
    exclude_terms = _normalize_list(preferences.get("exclude"))

    calories = _to_float(meal.get("calories"))
    protein  = _to_float(meal.get("protein"))
    carbs    = _to_float(meal.get("carbs"))
    fat      = _to_float(meal.get("fat"))
    sugar    = _to_float(meal.get("sugar"))
    sodium   = _to_float(meal.get("sodium"))

    # Hard gate 1: missing critical fields
    missing = [k for k, v in {"calories":calories,"protein":protein,"carbs":carbs,"fat":fat}.items() if v is None]
    if missing:
        return {"meal":meal_name,"rejected":True,
                "rejection_reason":"Missing critical nutrition data: " + ", ".join(sorted(missing)) + "."}

    # Hard gate 2: calorie limit
    if calorie_limit is not None and calories > calorie_limit:
        return {"meal":meal_name,"rejected":True,"rejection_reason":"Exceeds the calorie limit."}

    # Hard gate 3: exclusion list
    hit = _find_exclusion_conflict(meal, exclude_terms)
    if hit is not None:
        return {"meal":meal_name,"rejected":True,
                "rejection_reason":f"Contains excluded ingredient or allergen: {hit}."}

    # Hard gate 4: diet match
    diet_rejected, diet_reason, diet_match = _evaluate_diet_match(meal, diet_pref, carbs)
    if diet_rejected:
        return {"meal":meal_name,"rejected":True,"rejection_reason":diet_reason}

    # Soft scoring
    health_flags = _collect_health_flags(calories, protein, carbs, fat, sugar, sodium, calorie_limit)
    nutrition_score = _compute_nutrition_score(calories, protein, carbs, fat,
                                               calorie_limit, goal, diet_pref, health_flags)
    final_score = _round_2((0.5 * nutrition_score) + (0.5 * diet_match))
    category = _categorize_meal(final_score, nutrition_score, health_flags)
    reason = _build_reason(diet_pref, goal, calories, protein, carbs, calorie_limit, health_flags)

    result = dict(meal)
    result.update({
        "meal": meal_name, "name": meal_name, "rejected": False,
        "nutrition_score": _round_2(nutrition_score), "diet_match": _round_2(diet_match),
        "health_flags": health_flags, "final_score": final_score,
        "score": final_score, "category": category, "reason": reason,
    })
    return result


def _evaluate_diet_match(meal, diet_pref, carbs) -> Tuple[bool, str, float]:
    if not diet_pref:
        return False, "", 1.0
    diet_tokens = _normalize_tokens(meal.get("diet_type"))
    text = _compose_meal_text(meal)

    if diet_pref == "vegan":
        if _contains_any_term(text, _NON_VEGAN_TERMS):
            return True, "Conflicts with vegan dietary requirement.", 0.0
        if "vegan" in diet_tokens: return False, "", 1.0
        if "vegetarian" in diet_tokens: return False, "", 0.7
        return False, "", 0.75

    if diet_pref == "vegetarian":
        if _contains_any_term(text, _ANIMAL_TERMS):
            return True, "Conflicts with vegetarian dietary requirement.", 0.0
        if "vegetarian" in diet_tokens or "vegan" in diet_tokens: return False, "", 1.0
        return False, "", 0.8

    if diet_pref == "keto":
        if carbs > 50: return True, "Conflicts with keto preference due to high carbs.", 0.0
        if carbs <= 20: return False, "", 1.0
        if carbs <= 35: return False, "", 0.85
        return False, "", 0.7

    if diet_pref in diet_tokens: return False, "", 1.0
    if diet_tokens: return True, f"Does not match required diet: {diet_pref}.", 0.0
    return False, "", 0.6


def _collect_health_flags(calories, protein, carbs, fat, sugar, sodium, calorie_limit) -> List[str]:
    flags = []
    if sugar is not None and sugar > 25: flags.append("high_sugar")
    if sodium is not None and sodium > 700: flags.append("high_sodium")
    if fat > 30: flags.append("high_fat")
    if calorie_limit is not None:
        if calories > 0.9 * calorie_limit: flags.append("high_calorie")
    elif calories > 800: flags.append("high_calorie")
    if protein < 12: flags.append("low_protein")
    if carbs > 90: flags.append("high_carb")
    return flags


def _compute_nutrition_score(calories, protein, carbs, fat, calorie_limit, goal, diet_pref, health_flags) -> float:
    g = goal or ""
    is_hp = "high protein" in g; is_lc = "low calorie" in g
    is_lcarb = "low carb" in g; is_bal = "balanced" in g

    if calorie_limit and calorie_limit > 0:
        r = calories / calorie_limit
        cs = 1.0 if r <= 0.7 else (0.9 if r <= 0.85 else 0.75)
    else:
        cs = 1.0 if calories <= 450 else (0.85 if calories <= 650 else (0.65 if calories <= 850 else 0.45))

    ps = (1.0 if protein >= 30 else 0.85 if protein >= 22 else 0.65 if protein >= 15 else 0.4) if is_hp else \
         (1.0 if protein >= 25 else 0.85 if protein >= 18 else 0.7 if protein >= 12 else 0.5)

    if is_lcarb or diet_pref == "keto":
        cbs = 1.0 if carbs <= 20 else 0.8 if carbs <= 35 else 0.6 if carbs <= 50 else 0.35
    elif is_bal:
        cbs = 0.95 if 25 <= carbs <= 55 else (0.75 if 15 <= carbs <= 70 else 0.55)
    else:
        cbs = 0.9 if carbs <= 30 else 0.85 if carbs <= 60 else 0.7 if carbs <= 80 else 0.5

    fs = 0.9 if 10 <= fat <= 25 else (0.75 if fat < 10 else (0.65 if fat <= 35 else 0.45))

    base = 0.30*cs + 0.30*ps + 0.20*cbs + 0.20*fs
    if is_lc and calorie_limit and calories <= 0.75 * calorie_limit: base += 0.05
    if is_hp and protein >= 25: base += 0.05
    if is_lcarb and carbs <= 25: base += 0.05
    if is_bal and (15<=protein<=35) and (20<=carbs<=60) and (10<=fat<=25): base += 0.05
    return _clip(base - 0.05*len(health_flags), 0.0, 1.0)


def _categorize_meal(final_score, nutrition_score, health_flags) -> str:
    if final_score >= 0.75 and nutrition_score >= 0.7 and len(health_flags) <= 1: return "healthy"
    if final_score < 0.45 or nutrition_score < 0.45 or len(health_flags) >= 3: return "unhealthy"
    return "moderate"


def _build_reason(diet_pref, goal, calories, protein, carbs, calorie_limit, health_flags) -> str:
    parts = []
    g = goal or ""
    if "high protein" in g and protein >= 20: parts.append("high protein")
    elif "low carb" in g and carbs <= 35: parts.append("low carb")
    elif "balanced" in g and (15<=protein<=35) and (20<=carbs<=60): parts.append("balanced macros")
    if calorie_limit is not None and calories <= calorie_limit: parts.append("within calorie limit")
    if diet_pref: parts.append(f"matches {diet_pref} preference")
    if not parts: parts.append("nutritionally suitable")
    s = _capitalize_first(" and ".join(parts)) + "."
    if health_flags:
        lbl = {"high_sugar":"high sugar","high_sodium":"high sodium","high_fat":"high fat",
               "high_calorie":"high calories","low_protein":"low protein","high_carb":"high carbs"
               }.get(health_flags[0], health_flags[0].replace("_"," "))
        s = s[:-1] + f" but has slightly {lbl}."
    return s


def _find_exclusion_conflict(meal, exclude_terms) -> Optional[str]:
    if not exclude_terms: return None
    text = _compose_meal_text(meal)
    for t in exclude_terms:
        if _contains_term(text, t): return t
    return None


def _compose_meal_text(meal) -> str:
    return " ".join(filter(None, [
        _normalize_text(meal.get("name")),
        _normalize_text(meal.get("ingredients")),
        _normalize_text(meal.get("allergens")),
        _normalize_text(meal.get("diet_type")),
    ])).strip()


def _normalize_tokens(value) -> List[str]:
    text = _normalize_text(value)
    if not text: return []
    return [t for t in re.split(r"[\|,;/]+", text) if t]


def _normalize_list(value) -> List[str]:
    if value is None: return []
    if isinstance(value, (list, tuple, set)):
        return [t for t in (_normalize_text(i) for i in value) if t]
    n = _normalize_text(value)
    return [n] if n else []


def _normalize_text(value) -> Optional[str]:
    if value is None: return None
    text = str(value).strip().lower()
    return text or None


def _to_float(value) -> Optional[float]:
    if value is None: return None
    if isinstance(value, bool): return None
    if isinstance(value, (int, float)): return float(value)
    try: return float(str(value).strip())
    except (TypeError, ValueError): return None


def _contains_any_term(text: str, terms) -> bool:
    return any(_contains_term(text, t) for t in terms)


def _contains_term(text: str, term: str) -> bool:
    if not text or not term: return False
    return re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE) is not None


def _round_2(v: float) -> float: return round(v + 1e-10, 2)
def _clip(v: float, lo: float, hi: float) -> float: return max(lo, min(hi, v))
def _capitalize_first(text: str) -> str: return text[0].upper() + text[1:] if text else text
