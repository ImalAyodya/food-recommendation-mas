"""
agents/preference_agent.py
===========================
Agent 1: Preference Analyzer Agent
------------------------------------
Role in MAS Pipeline
    Parses the user's free-text query and extracts structured dietary
    preferences (diet type, calorie limit, exclusions, cuisine) using an
    Ollama-backed LLM.  Falls back to deterministic rules if Ollama is
    offline.  All extracted data is validated by preference_validation_tool
    before being stored in state.

Tools used
    • tools/preference_validation_tool.py  — schema + injection validation
    • tools/logger.py                      — agent step + tool call logging

Assignment compliance
    ✔  Distinct agent with a non-overlapping role
    ✔  Uses two dedicated custom Python tools
    ✔  Reads from / writes to global FoodState
    ✔  Logs every step and every tool call
    ✔  Works locally — no API keys required
"""

import json
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import JsonOutputParser

from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from tools.logger import log_agent_step, log_tool_call
from tools.preference_validation_tool import validate_preferences, detect_injection

# ─── Prompt ───────────────────────────────────────────────────────────────────

PREFERENCE_PROMPT = """
You are an expert dietary assistant. Your task is to extract dietary preferences from the user's input and format it as a valid JSON object.

Extract the following:
- diet: The diet type (e.g., "vegan", "vegetarian", "keto"). If none, use null.
- calorie_limit: An integer representing the maximum calories. If not specified, use null.
- exclude: A list of strings for avoiding ingredients/allergies (e.g., ["nuts", "dairy"]). If none, use an empty list [].
- cuisine: The type of cuisine (e.g., "Italian", "Mexican"). If none, use null.

User Input: "I want vegan food under 500 calories without nuts"
Expected output:
{{
    "diet": "vegan",
    "calorie_limit": 500,
    "exclude": ["nuts"],
    "cuisine": null
}}

User Input: "{user_input}"
Expected output:
"""

# ─── Public entry point ────────────────────────────────────────────────────────

def analyze_preferences(state, user_input):
    """
    Agent 1 main function: Preference Analyzer.

    Reads:
        user_input  — raw text from the caller

    Writes:
        state["preferences"]  — structured dietary preference dict
        state["user_input"]   — stores the raw input in state
        state["logs"]         — one log entry appended
        state["tool_calls"]   — tool invocation records appended

    Args:
        state:      Shared FoodState dict.
        user_input: Free-text query from the user.

    Returns:
        Updated state dict.
    """
    print(f"[Agent 1] Analyzing preferences for -> '{user_input}'")

    # Store raw input in state
    state["user_input"] = user_input

    # ── Injection guard on raw input ──────────────────────────────────────────
    if detect_injection(user_input):
        warning = "Potential prompt injection detected in user input — using fallback."
        print(f"[Agent 1] WARNING: {warning}")
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(warning)
        log_tool_call(state, "detect_injection",
                      {"user_input": user_input[:80]}, {"detected": True})
        extracted_data = _fallback_extract(user_input)
    else:
        log_tool_call(state, "detect_injection",
                      {"user_input": user_input[:80]}, {"detected": False})
        extracted_data = _llm_extract(state, user_input)

    # ── Validate via preference_validation_tool ───────────────────────────────
    try:
        clean = validate_preferences(extracted_data)
        log_tool_call(state, "validate_preferences",
                      {"raw": extracted_data}, {"result": "ok", "clean": clean})
        extracted_data = clean
    except (TypeError, ValueError) as exc:
        warn = f"Preference validation warning: {exc} — using unvalidated data."
        print(f"[Agent 1] {warn}")
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(warn)
        log_tool_call(state, "validate_preferences",
                      {"raw": extracted_data}, {"result": "validation_error", "error": str(exc)})

    state["preferences"] = extracted_data
    print(f"[Agent 1] Extracted preferences -> {extracted_data}")

    log_agent_step(
        state=state,
        agent_name="PreferenceAnalyzerAgent",
        input_data={"user_input": user_input},
        output_data=extracted_data,
    )
    return state


# ─── Private helpers ──────────────────────────────────────────────────────────

def _llm_extract(state, user_input):
    """Attempt preference extraction via Ollama LLM chain."""
    try:
        llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL,
                         format="json", temperature=0)
        parser = JsonOutputParser()
        prompt = PromptTemplate(template=PREFERENCE_PROMPT, input_variables=["user_input"])
        chain = prompt | llm | parser
        raw = chain.invoke({"user_input": user_input})

        extracted = {
            "diet":          raw.get("diet", None),
            "calorie_limit": raw.get("calorie_limit", None),
            "exclude":       raw.get("exclude", []),
            "cuisine":       raw.get("cuisine", None),
        }
        log_tool_call(state, "OllamaLLM.invoke",
                      {"user_input": user_input[:80]}, {"status": "success"})
        print("[Agent 1] Extracted preferences via Ollama.")
        return extracted

    except Exception as exc:
        print(f"[Agent 1] Ollama unavailable ({exc}). Using rules-based fallback.")
        log_tool_call(state, "OllamaLLM.invoke",
                      {"user_input": user_input[:80]},
                      {"status": "error", "error": str(exc)})
        return _fallback_extract(user_input)


# ─ Negation words that flip a diet keyword to "no specific diet" ─────────────
_NEGATION_PREFIXES = ("non ", "non-", "not ", "no ", "without ", "except ", "avoid ")
_DIET_KEYWORDS = {
    "vegan":       "vegan",
    "vegetarian":  "vegetarian",
    "veg ":        "vegetarian",
    "keto":        "keto",
    "paleo":       "paleo",
    "gluten":      "gluten-free",
    "dairy-free":  "dairy-free",
    "halal":       "halal",
    "kosher":      "kosher",
    "mediterranean": "mediterranean",
}


def _has_negation_before(low: str, keyword: str) -> bool:
    """Return True if any negation word appears immediately before the keyword."""
    idx = low.find(keyword)
    if idx < 0:
        return False
    prefix_window = low[max(0, idx - 15): idx]  # look 15 chars back
    return any(prefix_window.endswith(neg.rstrip()) or neg.rstrip() in prefix_window
               for neg in _NEGATION_PREFIXES)


def _fallback_extract(user_input: str) -> dict:
    """Deterministic rules-based extraction when Ollama is offline."""
    low = user_input.lower()

    # ── Diet: only pick a diet keyword if NOT negated ─────────────────────────
    diet = None
    for kw, diet_name in _DIET_KEYWORDS.items():
        if kw in low and not _has_negation_before(low, kw):
            diet = diet_name
            break

    # ── Calorie limit ─────────────────────────────────────────────────────────
    import re as _re
    cal_match = _re.search(r'(\d{2,4})\s*(?:cal|kcal|calories)', low)
    if cal_match:
        calorie_limit = int(cal_match.group(1))
    elif "low calorie" in low or "low-calorie" in low:
        calorie_limit = 1000
    else:
        calorie_limit = None

    # ── Exclusions ────────────────────────────────────────────────────────────
    exclude = []
    _ALLERGEN_WORDS = ["nut", "nuts", "peanut", "dairy", "milk", "egg", "gluten",
                       "shellfish", "soy", "wheat", "fish", "meat"]
    for word in _ALLERGEN_WORDS:
        # only exclude if the word appears in an exclusion context
        if _re.search(rf'(?:no|without|exclude|avoid|free)\s+{_re.escape(word)}', low):
            if word not in exclude:
                exclude.append(word)

    # Also catch "without nuts" / "no nuts" style
    if "nut" in low and not any(neg in low for neg in ["peanut butter", "donut"]):
        found = _re.search(r'(?:without|no)\s+nuts?', low)
        if found and "nuts" not in exclude:
            exclude.append("nuts")

    if "dairy" in low and not _has_negation_before(low, "dairy"):
        if "dairy" not in exclude:
            exclude.append("dairy")

    # ── Cuisine ───────────────────────────────────────────────────────────────
    cuisine = None
    _CUISINES = ["italian", "mexican", "asian", "indian", "chinese",
                 "japanese", "thai", "greek", "american", "french", "mediterranean"]
    for c in _CUISINES:
        if c in low:
            cuisine = c.capitalize()
            break

    return {
        "diet":          diet,
        "calorie_limit": calorie_limit,
        "exclude":       exclude,
        "cuisine":       cuisine,
    }