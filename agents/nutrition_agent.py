def analyze_nutrition(state):
    scored = []

    for meal in state["candidate_meals"]:
        score = 1 if meal["calories"] <= state["preferences"]["calorie_limit"] else 0.5
        meal["score"] = score
        scored.append(meal)

    state["scored_meals"] = scored
    return state