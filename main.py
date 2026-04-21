from agents.preference_agent import analyze_preferences
from agents.menu_agent import fetch_menu
from agents.nutrition_agent import analyze_nutrition
from agents.recommendation_agent import recommend_meals
from state import FoodState

def main():
    state: FoodState = {
        "preferences": {},
        "candidate_meals": [],
        "scored_meals": [],
        "final_recommendations": [],
        "logs": []
    }

    user_input = input("Enter your food preference: ")

    state = analyze_preferences(state, user_input)
    state = fetch_menu(state)
    state = analyze_nutrition(state)
    state = recommend_meals(state)

    print("\nTop Recommendations:")
    for meal in state["final_recommendations"]:
        print(f"  [{meal.get('rank', '?')}] {meal.get('name', 'Unknown')} "
              f"| Score: {meal.get('final_score', 0):.2f} "
              f"| {meal.get('calories', '?')} kcal "
              f"| {meal.get('category', '?').capitalize()}")

if __name__ == "__main__":
    main()