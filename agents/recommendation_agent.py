from config import TOP_N

def recommend_meals(state):
    sorted_meals = sorted(state["scored_meals"], key=lambda x: x["score"], reverse=True)
    state["final_recommendations"] = sorted_meals[:TOP_N]
    return state