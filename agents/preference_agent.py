def analyze_preferences(state, user_input):
    state["preferences"] = {
        "diet": "vegetarian" if "veg" in user_input else None,
        "calorie_limit": 500 if "low" in user_input else 1000
    }
    return state