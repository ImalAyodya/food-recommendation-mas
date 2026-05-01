"""
tests/test_observability.py
=============================
Tests for the logger/observability tool (tools/logger.py).

Covers:
    1. log_agent_step — appending records to state['logs']
    2. log_tool_call  — appending records to state['tool_calls']
    3. persist_trace  — writing trace JSON to disk

Run with:
    pytest tests/test_observability.py -v
"""

from __future__ import annotations

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.logger import log_agent_step, log_tool_call, persist_trace


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def blank_state():
    return {
        "user_input": "test input", "run_id": "run-abc",
        "preferences": {"diet": "vegan"}, "candidate_meals": [],
        "nutrition_evaluations": [], "scored_meals": [],
        "final_recommendations": [], "report_path": "/some/report.md",
        "json_path": "/some/result.json", "tool_calls": [], "errors": [], "logs": [],
    }


# ═════════════════════════════════════════════════════════════════════════════
# 1. log_agent_step
# ═════════════════════════════════════════════════════════════════════════════

class TestLogAgentStep:

    def test_appends_to_logs(self, blank_state):
        log_agent_step(blank_state, "AgentX", {"in": 1}, {"out": 2})
        assert len(blank_state["logs"]) == 1

    def test_log_has_required_keys(self, blank_state):
        log_agent_step(blank_state, "AgentX", {"in": 1}, {"out": 2})
        entry = blank_state["logs"][0]
        for key in ("agent", "input", "output", "timestamp"):
            assert key in entry, f"Missing key: {key}"

    def test_agent_name_stored(self, blank_state):
        log_agent_step(blank_state, "NutritionAnalyzerAgent", {}, {})
        assert blank_state["logs"][0]["agent"] == "NutritionAnalyzerAgent"

    def test_multiple_calls_accumulate(self, blank_state):
        for i in range(4):
            log_agent_step(blank_state, f"Agent{i}", {}, {})
        assert len(blank_state["logs"]) == 4

    def test_returns_state(self, blank_state):
        returned = log_agent_step(blank_state, "A", {}, {})
        assert returned is blank_state

    def test_initialises_missing_logs_key(self):
        state = {}  # no 'logs' key
        log_agent_step(state, "A", {}, {})
        assert "logs" in state and len(state["logs"]) == 1

    def test_timestamp_is_iso_format(self, blank_state):
        log_agent_step(blank_state, "A", {}, {})
        ts = blank_state["logs"][0]["timestamp"]
        # Should parse as an ISO datetime without raising
        from datetime import datetime
        datetime.fromisoformat(ts)


# ═════════════════════════════════════════════════════════════════════════════
# 2. log_tool_call
# ═════════════════════════════════════════════════════════════════════════════

class TestLogToolCall:

    def test_appends_to_tool_calls(self, blank_state):
        log_tool_call(blank_state, "batch_evaluate", {"n": 10}, {"accepted": 8})
        assert len(blank_state["tool_calls"]) == 1

    def test_entry_has_required_keys(self, blank_state):
        log_tool_call(blank_state, "rank_meals", {}, {})
        entry = blank_state["tool_calls"][0]
        for key in ("tool", "args", "result", "timestamp"):
            assert key in entry

    def test_tool_name_stored(self, blank_state):
        log_tool_call(blank_state, "validate_preferences", {}, {})
        assert blank_state["tool_calls"][0]["tool"] == "validate_preferences"

    def test_multiple_tool_calls_accumulate(self, blank_state):
        for tool in ["t1", "t2", "t3"]:
            log_tool_call(blank_state, tool, {}, {})
        assert len(blank_state["tool_calls"]) == 3

    def test_returns_state(self, blank_state):
        returned = log_tool_call(blank_state, "t", {}, {})
        assert returned is blank_state

    def test_initialises_missing_tool_calls_key(self):
        state = {}
        log_tool_call(state, "t", {}, {})
        assert "tool_calls" in state and len(state["tool_calls"]) == 1

    def test_args_and_result_stored(self, blank_state):
        args   = {"meal_count": 50}
        result = {"accepted": 12}
        log_tool_call(blank_state, "t", args, result)
        entry = blank_state["tool_calls"][0]
        assert entry["args"]   == args
        assert entry["result"] == result


# ═════════════════════════════════════════════════════════════════════════════
# 3. persist_trace
# ═════════════════════════════════════════════════════════════════════════════

class TestPersistTrace:

    def test_creates_file(self, blank_state, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.logger.LOGS_DIR", str(tmp_path))
        path = persist_trace(blank_state, "run-test-001")
        assert os.path.isfile(path)

    def test_file_extension_is_json(self, blank_state, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.logger.LOGS_DIR", str(tmp_path))
        path = persist_trace(blank_state, "run-abc")
        assert path.endswith(".json")

    def test_run_id_in_filename(self, blank_state, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.logger.LOGS_DIR", str(tmp_path))
        path = persist_trace(blank_state, "unique-run-xyz")
        assert "unique-run-xyz" in os.path.basename(path)

    def test_trace_json_is_parseable(self, blank_state, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.logger.LOGS_DIR", str(tmp_path))
        path = persist_trace(blank_state, "run-parse")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_trace_has_required_keys(self, blank_state, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.logger.LOGS_DIR", str(tmp_path))
        path = persist_trace(blank_state, "run-keys")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for key in ("run_id", "created_at", "user_input", "preferences",
                    "summary", "errors", "tool_calls", "logs"):
            assert key in data, f"Missing trace key: {key}"

    def test_trace_run_id_matches(self, blank_state, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.logger.LOGS_DIR", str(tmp_path))
        path = persist_trace(blank_state, "my-run-999")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["run_id"] == "my-run-999"

    def test_trace_summary_counts_are_correct(self, blank_state, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.logger.LOGS_DIR", str(tmp_path))
        blank_state["final_recommendations"] = [{"name": "A"}, {"name": "B"}]
        path = persist_trace(blank_state, "run-summary")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["summary"]["final_recommendations"] == 2

    def test_creates_directory_if_missing(self, blank_state, tmp_path, monkeypatch):
        new_dir = str(tmp_path / "nested" / "logs")
        monkeypatch.setattr("tools.logger.LOGS_DIR", new_dir)
        path = persist_trace(blank_state, "run-mkdir")
        assert os.path.isdir(new_dir)
        assert os.path.isfile(path)

    def test_report_path_stored_in_trace(self, blank_state, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.logger.LOGS_DIR", str(tmp_path))
        blank_state["report_path"] = "/outputs/reports/test.md"
        path = persist_trace(blank_state, "run-rp")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["report_path"] == "/outputs/reports/test.md"
