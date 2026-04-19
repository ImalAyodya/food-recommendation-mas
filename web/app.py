import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import json
import datetime

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# ─── Import the MAS pipeline ───────────────────────────────────────────────────
from agents.preference_agent import analyze_preferences
from agents.menu_agent import fetch_menu
from agents.nutrition_agent import analyze_nutrition
from agents.recommendation_agent import recommend_meals
from state import FoodState


def make_state() -> FoodState:
    return {
        "preferences": {},
        "candidate_meals": [],
        "scored_meals": [],
        "final_recommendations": [],
        "logs": []
    }


# ─── HTML entry point ─────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


# ─── API: Run the full pipeline ────────────────────────────────────────────────
@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json(force=True)
    user_input = data.get('query', '').strip()

    if not user_input:
        return jsonify({'error': 'Please provide a food preference query.'}), 400

    state = make_state()
    pipeline_steps = []

    # ── Step 1: Preference Analysis ──────────────────────────────────────────
    t0 = datetime.datetime.now()
    state = analyze_preferences(state, user_input)
    t1 = datetime.datetime.now()
    pipeline_steps.append({
        'agent': 'PreferenceAnalyzerAgent',
        'icon': '🧠',
        'status': 'done',
        'duration_ms': int((t1 - t0).total_seconds() * 1000),
        'output': state['preferences']
    })

    # ── Step 2: Menu Fetching ────────────────────────────────────────────────
    t0 = datetime.datetime.now()
    try:
        state = fetch_menu(state)
        menu_count = len(state['candidate_meals'])
        menu_status = 'done'
    except Exception as e:
        menu_count = 0
        menu_status = 'error'
        state['candidate_meals'] = []
    t1 = datetime.datetime.now()
    pipeline_steps.append({
        'agent': 'MenuFetcherAgent',
        'icon': '🍽️',
        'status': menu_status,
        'duration_ms': int((t1 - t0).total_seconds() * 1000),
        'output': {'candidate_count': menu_count}
    })

    # ── Step 3: Nutrition Analysis ───────────────────────────────────────────
    t0 = datetime.datetime.now()
    calorie_limit = state['preferences'].get('calorie_limit')
    if not calorie_limit:
        # Default calorie limit if not specified
        state['preferences']['calorie_limit'] = 9999

    if state['candidate_meals']:
        state = analyze_nutrition(state)
    else:
        state['scored_meals'] = []
    t1 = datetime.datetime.now()
    pipeline_steps.append({
        'agent': 'NutritionAnalyzerAgent',
        'icon': '📊',
        'status': 'done',
        'duration_ms': int((t1 - t0).total_seconds() * 1000),
        'output': {'scored_count': len(state['scored_meals'])}
    })

    # ── Step 4: Recommendation ───────────────────────────────────────────────
    t0 = datetime.datetime.now()
    state = recommend_meals(state)
    t1 = datetime.datetime.now()
    pipeline_steps.append({
        'agent': 'RecommendationAgent',
        'icon': '⭐',
        'status': 'done',
        'duration_ms': int((t1 - t0).total_seconds() * 1000),
        'output': {'recommendation_count': len(state['final_recommendations'])}
    })

    # ── Build recommendation cards ───────────────────────────────────────────
    recommendations = []
    for meal in state['final_recommendations']:
        rec = {
            'name': meal.get('name', 'Unknown'),
            'calories': _safe_float(meal.get('calories', 0)),
            'protein': _safe_float(meal.get('protein', 0)),
            'fat': _safe_float(meal.get('fat', 0)),
            'carbs': _safe_float(meal.get('carbs', 0)),
            'diet_type': meal.get('diet_type', ''),
            'cuisine': meal.get('cuisine', ''),
            'allergens': meal.get('allergens', 'none'),
            'ingredients': meal.get('ingredients', ''),
            'score': _safe_float(meal.get('score', 0))
        }
        recommendations.append(rec)

    return jsonify({
        'query': user_input,
        'preferences': state['preferences'],
        'pipeline': pipeline_steps,
        'recommendations': recommendations,
        'logs': state.get('logs', [])
    })


# ─── API: Get system status ───────────────────────────────────────────────────
@app.route('/api/status', methods=['GET'])
def status():
    """Check if Ollama is reachable and dataset is loaded."""
    import requests as req
    ollama_ok = False
    try:
        r = req.get('http://localhost:11434', timeout=2)
        ollama_ok = r.status_code == 200
    except Exception:
        pass

    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed', 'meals.csv')
    dataset_ok = os.path.exists(data_path)
    row_count = 0
    if dataset_ok:
        with open(data_path, 'r', encoding='utf-8') as f:
            row_count = sum(1 for _ in f) - 1  # minus header

    return jsonify({
        'ollama': ollama_ok,
        'dataset': dataset_ok,
        'dataset_rows': row_count,
        'model': 'llama3.2'
    })


def _safe_float(val):
    try:
        return round(float(val), 1)
    except (TypeError, ValueError):
        return 0.0


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print("\nFood Recommendation MAS - Web UI")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)
