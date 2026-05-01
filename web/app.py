import sys
import os
import uuid
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# ─── Import LangGraph workflow ────────────────────────────────────────────────
from workflow import build_graph, make_initial_state

# Pre-compile the graph once at startup (avoids recompilation on every request)
_graph = build_graph()


# ─── HTML entry point ─────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


# ─── API: Run the full pipeline via LangGraph ─────────────────────────────────
@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json(force=True)
    user_input = data.get('query', '').strip()

    if not user_input:
        return jsonify({'error': 'Please provide a food preference query.'}), 400

    run_id = str(uuid.uuid4())
    state  = make_initial_state(user_input, run_id=run_id)


    try:
        t_start = datetime.datetime.now()
        result  = _graph.invoke(state)

        # ── Calorie-relaxation retry ──────────────────────────────────────────
        # If the pipeline returns 0 recommendations AND the user specified a
        # very tight calorie limit, automatically retry with 1.5× the limit
        # so the user always sees *something* useful.
        original_cal = result.get("preferences", {}).get("calorie_limit")
        if (
            len(result.get("final_recommendations", [])) == 0
            and original_cal is not None
            and original_cal < 600
        ):
            relaxed_cal = round(original_cal * 1.5)
            retry_state = make_initial_state(user_input, run_id=run_id)
            retry_state["errors"].append(
                f"No meals found under {int(original_cal)} kcal. "
                f"Showing results for up to {relaxed_cal} kcal instead."
            )
            result = _graph.invoke(retry_state)
            # Override the calorie_limit in preferences for display purposes
            result.setdefault("preferences", {})["calorie_limit"] = relaxed_cal
            result.setdefault("preferences", {})["calorie_relaxed"] = True

        t_total = int((datetime.datetime.now() - t_start).total_seconds() * 1000)
    except Exception as exc:
        return jsonify({'error': f'Pipeline error: {exc}'}), 500


    # ── Rebuild per-agent pipeline steps from logs ────────────────────────────
    AGENT_ICONS = {
        "PreferenceAnalyzerAgent":    "🧠",
        "MenuFetcherAgent":           "🍽️",
        "NutritionAnalyzerAgent":     "📊",
        "RecommendationReportAgent":  "⭐",
    }
    pipeline_steps = []
    for log in result.get("logs", []):
        agent = log.get("agent", "Unknown")
        pipeline_steps.append({
            "agent":  agent,
            "icon":   AGENT_ICONS.get(agent, "🤖"),
            "status": "done",
            "output": log.get("output", {}),
        })

    # ── Build recommendation cards ────────────────────────────────────────────
    recommendations = []
    for meal in result.get("final_recommendations", []):
        recommendations.append({
            "name":       meal.get("name", "Unknown"),
            "calories":   _safe_float(meal.get("calories", 0)),
            "protein":    _safe_float(meal.get("protein", 0)),
            "fat":        _safe_float(meal.get("fat", 0)),
            "carbs":      _safe_float(meal.get("carbs", 0)),
            "diet_type":  meal.get("diet_type", ""),
            "cuisine":    meal.get("cuisine", ""),
            "allergens":  meal.get("allergens", "none"),
            "ingredients":meal.get("ingredients", ""),
            "score":      _safe_float(meal.get("final_score", 0)),
            "rank":       meal.get("rank", 0),
            "badge":      meal.get("badge", ""),
            "category":   meal.get("category", ""),
            "reason":     meal.get("reason", ""),
        })

    return jsonify({
        "run_id":          run_id,
        "query":           user_input,
        "preferences":     result.get("preferences", {}),
        "pipeline":        pipeline_steps,
        "recommendations": recommendations,
        "report_path":     result.get("report_path"),
        "json_path":       result.get("json_path"),
        "tool_calls":      len(result.get("tool_calls", [])),
        "errors":          result.get("errors", []),
        "logs":            result.get("logs", []),
        "total_ms":        t_total,
    })


# ─── API: System status ───────────────────────────────────────────────────────
@app.route('/api/status', methods=['GET'])
def status():
    """Check if Ollama is reachable and the dataset is loaded."""
    import requests as req
    ollama_ok = False
    try:
        r = req.get('http://localhost:11434', timeout=2)
        ollama_ok = r.status_code == 200
    except Exception:
        pass

    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'processed', 'meals.csv'
    )
    dataset_ok = os.path.exists(data_path)
    row_count  = 0
    if dataset_ok:
        with open(data_path, 'r', encoding='utf-8') as f:
            row_count = sum(1 for _ in f) - 1  # minus header

    return jsonify({
        'ollama':       ollama_ok,
        'dataset':      dataset_ok,
        'dataset_rows': row_count,
        'model':        'llama3.2',
    })


# ─── API: Download report ─────────────────────────────────────────────────────
@app.route('/api/report/<run_id>', methods=['GET'])
def get_report(run_id: str):
    """Return the Markdown report for a given run_id (basic file lookup)."""
    from flask import send_file
    reports_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'outputs', 'reports'
    )
    for fname in os.listdir(reports_dir) if os.path.isdir(reports_dir) else []:
        if run_id in fname and fname.endswith('.md'):
            return send_file(os.path.join(reports_dir, fname),
                             mimetype='text/markdown', as_attachment=True)
    return jsonify({'error': 'Report not found'}), 404


def _safe_float(val):
    try:
        return round(float(val), 1)
    except (TypeError, ValueError):
        return 0.0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print("\nFood Recommendation MAS — Web UI")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)
