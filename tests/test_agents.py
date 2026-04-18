def test_preference():
    from agents.preference_agent import analyze_preferences

    state = {"preferences": {}, "candidate_meals": [], "scored_meals": [], "final_recommendations": [], "logs": []}
    result = analyze_preferences(state, "low calorie veg food")

    assert result["preferences"]["calorie_limit"] == 500