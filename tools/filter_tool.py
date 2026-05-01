from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional
import re

import pandas as pd


def filter_meals(df: pd.DataFrame, preferences: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """
    Filter and clean meals from the dataset based on structured user preferences.

    Args:
        df: Pandas DataFrame containing the meals dataset.
        preferences: Mapping with optional keys:
            - diet (str or None)
            - calorie_limit (int/float or None)
            - exclude (list[str])
            - cuisine (str or None)
            - prep_time / prep_time_min / max_prep_time (int/float or None)

    Returns:
        A deterministic list of cleaned meal dictionaries suitable for downstream agents.

    Raises:
        TypeError: If inputs are of the wrong type.
        ValueError: If required columns are missing or preferences are invalid.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(preferences, Mapping):
        raise TypeError("preferences must be a mapping")

    required_columns = {"name", "ingredients", "calories", "diet_type", "cuisine", "allergens"}
    missing = sorted(col for col in required_columns if col not in df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    working = df.copy()

    for col in ["name", "ingredients", "diet_type", "cuisine", "allergens"]:
        working[col] = working[col].astype(str).fillna("").str.strip()

    # Remove incomplete records
    working = working[(working["name"] != "") & (working["ingredients"] != "")]

    for col in ["calories", "protein", "fat", "carbs"]:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")

    working = working[working["calories"].notna() & (working["calories"] > 0)]

    diet = preferences.get("diet")
    if diet is not None:
        if not isinstance(diet, str):
            raise ValueError("preferences['diet'] must be a string or None")
        diet = diet.strip()
        if diet:
            # ── Soft diet filter ───────────────────────────────────────────────
            # Try an exact pipe-delimited token match first (strictest)
            token = re.escape(diet)
            strict_pattern = rf"(?:^|\|)\s*{token}\s*(?:\||$)"
            strict_match = working[working["diet_type"].str.contains(
                strict_pattern, case=False, na=False)]

            if len(strict_match) >= 1:
                working = strict_match
            else:
                # Fall back to a substring match (looser)
                loose_match = working[working["diet_type"].str.contains(
                    re.escape(diet), case=False, na=False)]
                if len(loose_match) >= 1:
                    working = loose_match
                # else: diet filter skipped entirely — let nutrition agent handle
                # diet compliance via ingredient analysis instead

    cuisine = preferences.get("cuisine")
    if cuisine is not None:
        if not isinstance(cuisine, str):
            raise ValueError("preferences['cuisine'] must be a string or None")
        cuisine = cuisine.strip()
        if cuisine:
            # ── Soft cuisine filter ────────────────────────────────────────────
            cuisine_match = working[working["cuisine"].str.contains(
                re.escape(cuisine), case=False, na=False)]
            if len(cuisine_match) >= 1:
                working = cuisine_match
            # else: skip cuisine filter rather than returning 0 results

    calorie_limit = preferences.get("calorie_limit")
    if calorie_limit is not None:
        try:
            limit = float(calorie_limit)
        except (TypeError, ValueError):
            raise ValueError("preferences['calorie_limit'] must be a number or None")
        if limit <= 0:
            raise ValueError("preferences['calorie_limit'] must be greater than 0")
        working = working[working["calories"] <= limit]

    exclude_raw = preferences.get("exclude", [])
    if exclude_raw is None:
        exclude_items: List[str] = []
    elif isinstance(exclude_raw, (list, tuple, set)):
        exclude_items = [str(item).strip().lower() for item in exclude_raw if str(item).strip()]
    else:
        raise ValueError("preferences['exclude'] must be a list of strings")

    for item in exclude_items:
        combined = (
            working["ingredients"].fillna("").astype(str)
            + "|"
            + working["allergens"].fillna("").astype(str)
        ).str.lower()
        working = working[~combined.str.contains(re.escape(item), na=False)]

    prep_time = _get_prep_time_limit(preferences)
    if prep_time is not None:
        time_col = _pick_prep_time_column(working)
        if time_col is None:
            raise ValueError("Dataset does not include a prep time column for filtering.")
        working[time_col] = pd.to_numeric(working[time_col], errors="coerce")
        working = working[working[time_col].notna() & (working[time_col] <= prep_time)]

    base_columns = [
        "name",
        "calories",
        "protein",
        "fat",
        "carbs",
        "diet_type",
        "cuisine",
        "allergens",
        "ingredients",
        "prep_time_min",
        "total_time_min",
        "prep_time",
    ]
    output_columns = [col for col in base_columns if col in working.columns]
    return working[output_columns].to_dict(orient="records")


def _get_prep_time_limit(preferences: Mapping[str, Any]) -> Optional[float]:
    prep_time = None
    for key in ("prep_time", "prep_time_min", "max_prep_time"):
        if key in preferences and preferences[key] is not None:
            prep_time = preferences[key]
            break
    if prep_time is None:
        return None
    try:
        limit = float(prep_time)
    except (TypeError, ValueError):
        raise ValueError("preferences['prep_time'] must be a number or None")
    if limit <= 0:
        raise ValueError("preferences['prep_time'] must be greater than 0")
    return limit


def _pick_prep_time_column(df: pd.DataFrame) -> Optional[str]:
    for col in ("prep_time_min", "total_time_min", "prep_time"):
        if col in df.columns:
            return col
    return None
