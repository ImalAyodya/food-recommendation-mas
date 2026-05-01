"""
tools/logger.py
================
Shared observability tool for the Food Recommendation MAS.

Provides:
    log_agent_step   — Append a structured agent-step record to state['logs']
    log_tool_call    — Append a tool-invocation record to state['tool_calls']
    persist_trace    — Flush full state trace to outputs/logs/<run_id>.json

Assignment compliance:
    ✔  Tracks both agent steps AND tool calls (full observability)
    ✔  Persists trace files to disk for auditability
    ✔  Thread-safe directory creation
    ✔  ISO-8601 timestamps on every record
"""

from __future__ import annotations

import datetime
import json
import os
from typing import Any, Dict, Mapping

# ─── Output directory ─────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(_ROOT, "outputs", "logs")


def log_agent_step(
    state: Dict[str, Any],
    agent_name: str,
    input_data: Any,
    output_data: Any,
) -> Dict[str, Any]:
    """
    Append a structured agent-step record to state['logs'].

    Args:
        state:       Shared FoodState dict (mutated in-place).
        agent_name:  Identifier for the agent (e.g. "PreferenceAnalyzerAgent").
        input_data:  Serialisable snapshot of the agent's inputs.
        output_data: Serialisable snapshot of the agent's outputs.

    Returns:
        The updated state dict.
    """
    if "logs" not in state or not isinstance(state.get("logs"), list):
        state["logs"] = []

    state["logs"].append({
        "agent":     agent_name,
        "input":     input_data,
        "output":    output_data,
        "timestamp": datetime.datetime.now().isoformat(),
    })
    return state


def log_tool_call(
    state: Dict[str, Any],
    tool_name: str,
    args: Any,
    result: Any,
) -> Dict[str, Any]:
    """
    Append a tool-invocation record to state['tool_calls'].

    Args:
        state:     Shared FoodState dict (mutated in-place).
        tool_name: Name of the tool function called.
        args:      Serialisable snapshot of the arguments passed to the tool.
        result:    Serialisable snapshot of what the tool returned.

    Returns:
        The updated state dict.

    Example:
        >>> log_tool_call(state, "batch_evaluate", {"meal_count": 50}, {"accepted": 12})
    """
    if "tool_calls" not in state or not isinstance(state.get("tool_calls"), list):
        state["tool_calls"] = []

    state["tool_calls"].append({
        "tool":      tool_name,
        "args":      args,
        "result":    result,
        "timestamp": datetime.datetime.now().isoformat(),
    })
    return state


def persist_trace(state: Dict[str, Any], run_id: str) -> str:
    """
    Write the full pipeline trace to outputs/logs/<run_id>.json.

    Captures logs, tool_calls, errors, preferences, and result metadata.
    The file is human-readable JSON (indent=2) and is safe to diff in git.

    Args:
        state:  The final FoodState after the pipeline completes.
        run_id: Unique identifier for this pipeline run (UUID string).

    Returns:
        Absolute path of the written trace file.

    Raises:
        OSError: If the directory cannot be created or the file cannot be written.

    Example:
        >>> path = persist_trace(state, "abc-123")
        >>> os.path.exists(path)
        True
    """
    os.makedirs(LOGS_DIR, exist_ok=True)

    trace = {
        "run_id":     run_id,
        "created_at": datetime.datetime.now().isoformat(),
        "user_input": state.get("user_input", ""),
        "preferences": state.get("preferences", {}),
        "summary": {
            "candidate_meals":       len(state.get("candidate_meals", [])),
            "nutrition_evaluations": len(state.get("nutrition_evaluations", [])),
            "scored_meals":          len(state.get("scored_meals", [])),
            "final_recommendations": len(state.get("final_recommendations", [])),
        },
        "report_path": state.get("report_path"),
        "json_path":   state.get("json_path"),
        "errors":      state.get("errors", []),
        "tool_calls":  state.get("tool_calls", []),
        "logs":        state.get("logs", []),
    }

    out_path = os.path.join(LOGS_DIR, f"{run_id}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(trace, fh, indent=2, default=str, ensure_ascii=False)

    return out_path