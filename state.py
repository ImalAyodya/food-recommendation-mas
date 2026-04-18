from typing import TypedDict, List, Dict, Any

class FoodState(TypedDict):
    preferences: dict
    candidate_meals: List[Dict]
    scored_meals: List[Dict]
    final_recommendations: List[Dict]
    logs: List[Dict[str, Any]]