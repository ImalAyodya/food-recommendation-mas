from typing import Dict, Any, List, Mapping, Optional
import json

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from config import DATA_PATH, OLLAMA_BASE_URL, OLLAMA_MODEL
from state import FoodState
from tools.data_loader import load_data
from tools.filter_tool import filter_meals
from tools.logger import log_agent_step

MenuFetcherState = FoodState
# LLM is only used to normalize structured preferences into a strict JSON schema.
llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, format="json")

# System prompt focused on validation/normalization, not meal generation.
MENU_FETCHER_PROMPT = """
You are the Menu Fetcher Agent in a Food Recommendation Multi-Agent System.

Responsibilities:
- Validate and normalize the structured preferences from the Preference Analyzer.
- Do NOT invent new constraints or recipes.
- Keep the same semantic meaning as the input.
- Output only valid JSON with the schema below.

Constraints:
- If the input contains a value, preserve it (normalize formatting only).
- If a value is unknown, use null.
- Exclude must be a list of strings (possibly empty).
- calorie_limit and prep_time must be numbers or null.
- Keep diet/cuisine as short tags (e.g., "vegan", "asian").

Input preferences (JSON):
{preferences_json}

Example (preserve provided values):
Input:
{{"diet": "vegan", "calorie_limit": 500, "exclude": ["nuts"], "cuisine": "asian"}}
Output:
{{"diet": "vegan", "calorie_limit": 500, "exclude": ["nuts"], "cuisine": "asian", "prep_time": null}}

Output JSON schema:
{{
  "diet": "vegan",
  "calorie_limit": 500,
  "exclude": ["nuts"],
  "cuisine": "asian",
  "prep_time": 30
}}
"""


def fetch_menu(state: MenuFetcherState) -> MenuFetcherState:
    """
    LangGraph-compatible menu retrieval node that filters meals based on preferences.
    """
    if "preferences" not in state or not isinstance(state["preferences"], dict):
        raise ValueError("State must include a 'preferences' dictionary")

    preferences: Dict[str, Any] = state["preferences"]
    try:
        print(f"Agent 2: Menu Fetcher received preferences -> {preferences}")
        # Normalize user preferences with the LLM so downstream filtering is deterministic.
        normalized_preferences = _normalize_preferences_with_llm(preferences)
        normalized_preferences = _merge_with_fallback(preferences, normalized_preferences)
        print(f"Agent 2: Normalized preferences -> {normalized_preferences}")
        # Load dataset and filter against the normalized constraints.
        df = load_data(DATA_PATH)
        candidate_meals = filter_meals(df, normalized_preferences)
        print(f"Agent 2: Candidate meals found -> {len(candidate_meals)}")

        state["preferences"] = normalized_preferences
        state["candidate_meals"] = candidate_meals

        # Log success for UI/pipeline observability.
        log_agent_step(
            state=state,
            agent_name="MenuFetcherAgent",
            input_data={"preferences": preferences},
            output_data={
                "normalized_preferences": normalized_preferences,
                "candidate_count": len(candidate_meals),
                "status": "success",
            },
        )
        return state
    except Exception as exc:
        # Fail fast but still emit a log entry for debugging.
        state["candidate_meals"] = []
        print(f"Agent 2: Menu Fetcher failed -> {exc}")
        log_agent_step(
            state=state,
            agent_name="MenuFetcherAgent",
            input_data={"preferences": preferences},
            output_data={
                "status": "error",
                "error": str(exc),
            },
        )
        raise


def _normalize_preferences_with_llm(preferences: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Normalize preference fields using the LLM to a strict, typed schema.
    """
    if not isinstance(preferences, Mapping):
        raise TypeError("preferences must be a mapping")

    parser = JsonOutputParser()
    prompt = PromptTemplate(template=MENU_FETCHER_PROMPT, input_variables=["preferences_json"])
    chain = prompt | llm | parser

    normalized = chain.invoke({"preferences_json": json.dumps(preferences, ensure_ascii=False)})

    return {
        "diet": _coerce_optional_str(normalized.get("diet")),
        "calorie_limit": _coerce_optional_number(normalized.get("calorie_limit")),
        "exclude": _coerce_str_list(normalized.get("exclude")),
        "cuisine": _coerce_optional_str(normalized.get("cuisine")),
        "prep_time": _coerce_optional_number(normalized.get("prep_time")),
    }


def _merge_with_fallback(
    original: Mapping[str, Any],
    normalized: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Preserve original non-empty values when the LLM returns null/empty fields.
    """
    merged: Dict[str, Any] = dict(normalized)

    if merged.get("diet") in (None, "") and original.get("diet"):
        merged["diet"] = original.get("diet")
    if merged.get("cuisine") in (None, "") and original.get("cuisine"):
        merged["cuisine"] = original.get("cuisine")
    if merged.get("calorie_limit") is None and original.get("calorie_limit") is not None:
        merged["calorie_limit"] = original.get("calorie_limit")

    original_exclude = original.get("exclude")
    if (not merged.get("exclude")) and isinstance(original_exclude, list) and original_exclude:
        merged["exclude"] = original_exclude

    if merged.get("prep_time") is None and original.get("prep_time") is not None:
        merged["prep_time"] = original.get("prep_time")

    return merged


def _coerce_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Expected a string or null for textual preference fields")
    cleaned = value.strip()
    return cleaned or None


def _coerce_optional_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Expected a number or null for numeric preference fields")
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("Expected a number or null for numeric preference fields") from exc


def _coerce_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Expected a list of strings for exclude")
    cleaned: List[str] = []
    for item in value:
        if item is None:
            continue
        if not isinstance(item, str):
            raise ValueError("Expected a list of strings for exclude")
        entry = item.strip()
        if entry:
            cleaned.append(entry)
    return cleaned
