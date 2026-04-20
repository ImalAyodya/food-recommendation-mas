import pandas as pd

from tools.filter_tool import filter_meals


def test_filter_meals_applies_preferences_and_cleans_data():
    df = pd.DataFrame(
        [
            {
                "name": "Vegan Salad",
                "ingredients": "lettuce|100g; tomato|50g",
                "calories": 320,
                "protein": 18,
                "fat": 5,
                "carbs": 30,
                "diet_type": "vegan|vegetarian",
                "cuisine": "Asian",
                "allergens": "none",
            },
            {
                "name": "Nut Curry",
                "ingredients": "peanut|10g; rice|100g",
                "calories": 450,
                "protein": 12,
                "fat": 20,
                "carbs": 40,
                "diet_type": "vegan",
                "cuisine": "Asian",
                "allergens": "nuts",
            },
            {
                "name": "Cheese Pasta",
                "ingredients": "cheese|50g; pasta|100g",
                "calories": 600,
                "protein": 20,
                "fat": 25,
                "carbs": 70,
                "diet_type": "vegetarian",
                "cuisine": "Italian",
                "allergens": "dairy",
            },
            {
                "name": "",
                "ingredients": "spinach|10g",
                "calories": 100,
                "protein": 2,
                "fat": 1,
                "carbs": 5,
                "diet_type": "vegan",
                "cuisine": "Asian",
                "allergens": "none",
            },
            {
                "name": "Bad Calories",
                "ingredients": "rice|10g",
                "calories": "not-a-number",
                "protein": 2,
                "fat": 1,
                "carbs": 5,
                "diet_type": "vegan",
                "cuisine": "Asian",
                "allergens": "none",
            },
        ]
    )

    preferences = {
        "diet": "vegan",
        "calorie_limit": 500,
        "exclude": ["nuts"],
        "cuisine": "asian",
    }

    results = filter_meals(df, preferences)

    assert len(results) == 1
    assert results[0]["name"] == "Vegan Salad"
