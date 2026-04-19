import pytest
from agents.preference_agent import analyze_preferences

@pytest.fixture
def base_state():
    return {
        "preferences": {},
        "candidate_meals": [],
        "scored_meals": [],
        "final_recommendations": [],
        "logs": []
    }

def test_extract_vegan_under_500(mocker, base_state):
    # Mocking the langchain invoke to simulate Ollama's response
    mock_invoke = mocker.patch("langchain_core.runnables.RunnableSequence.invoke")
    mock_invoke.return_value = {
        "diet": "vegan",
        "calorie_limit": 500,
        "exclude": [],
        "cuisine": None
    }
    
    result = analyze_preferences(base_state, "I want vegan food under 500 calories")
    assert result["preferences"]["diet"] == "vegan"
    assert result["preferences"]["calorie_limit"] == 500
    assert result["preferences"]["exclude"] == []
    assert result["preferences"]["cuisine"] is None
    assert len(result["logs"]) == 1
    assert result["logs"][0]["agent"] == "PreferenceAnalyzerAgent"

def test_extract_vegetarian_no_nuts(mocker, base_state):
    mock_invoke = mocker.patch("langchain_core.runnables.RunnableSequence.invoke")
    mock_invoke.return_value = {
        "diet": "vegetarian",
        "calorie_limit": None,
        "exclude": ["nuts"],
        "cuisine": None
    }
    
    result = analyze_preferences(base_state, "I want vegetarian food without nuts")
    assert result["preferences"]["diet"] == "vegetarian"
    assert result["preferences"]["exclude"] == ["nuts"]

def test_extract_keto_high_protein(mocker, base_state):
    mock_invoke = mocker.patch("langchain_core.runnables.RunnableSequence.invoke")
    mock_invoke.return_value = {
        "diet": "keto",
        "calorie_limit": 800,
        "exclude": [],
        "cuisine": None
    }
    
    result = analyze_preferences(base_state, "Give me a keto meal under 800 cals")
    assert result["preferences"]["diet"] == "keto"

def test_extract_italian_cuisine(mocker, base_state):
    mock_invoke = mocker.patch("langchain_core.runnables.RunnableSequence.invoke")
    mock_invoke.return_value = {
        "diet": None,
        "calorie_limit": None,
        "exclude": [],
        "cuisine": "Italian"
    }
    
    result = analyze_preferences(base_state, "I am craving Italian tonight")
    assert result["preferences"]["cuisine"] == "Italian"

def test_extract_multiple_allergies(mocker, base_state):
    mock_invoke = mocker.patch("langchain_core.runnables.RunnableSequence.invoke")
    mock_invoke.return_value = {
        "diet": None,
        "calorie_limit": None,
        "exclude": ["peanuts", "shellfish"],
        "cuisine": None
    }
    
    result = analyze_preferences(base_state, "Don't include peanuts or shellfish")
    assert "peanuts" in result["preferences"]["exclude"]
    assert "shellfish" in result["preferences"]["exclude"]

def test_extract_empty_string(mocker, base_state):
    mock_invoke = mocker.patch("langchain_core.runnables.RunnableSequence.invoke")
    mock_invoke.return_value = {
        "diet": None,
        "calorie_limit": None,
        "exclude": [],
        "cuisine": None
    }
    
    result = analyze_preferences(base_state, "")
    assert result["preferences"]["diet"] is None
    print(result["preferences"])

# --- Fallback Logic Tests (When Ollama isn't running) ---

def test_fallback_vegan_500(mocker, base_state):
    # Force an exception to trigger the fallback logic
    mock_invoke = mocker.patch("langchain_core.runnables.RunnableSequence.invoke", side_effect=Exception("Ollama Offline"))
    result = analyze_preferences(base_state, "I want vegan food 500 calories without nuts")
    assert result["preferences"]["diet"] == "vegan"
    assert result["preferences"]["calorie_limit"] == 500
    assert "nuts" in result["preferences"]["exclude"]

def test_fallback_vegetarian(mocker, base_state):
    mocker.patch("langchain_core.runnables.RunnableSequence.invoke", side_effect=Exception("Ollama Offline"))
    result = analyze_preferences(base_state, "veg food")
    assert result["preferences"]["diet"] == "vegetarian"

def test_fallback_mexican_cuisine(mocker, base_state):
    mocker.patch("langchain_core.runnables.RunnableSequence.invoke", side_effect=Exception("Ollama Offline"))
    result = analyze_preferences(base_state, "some mexican stuff")
    assert result["preferences"]["cuisine"] == "Mexican"

def test_fallback_dairy_allergy(mocker, base_state):
    mocker.patch("langchain_core.runnables.RunnableSequence.invoke", side_effect=Exception("Ollama Offline"))
    result = analyze_preferences(base_state, "no dairy, low calorie")
    assert "dairy" in result["preferences"]["exclude"]
    assert result["preferences"]["calorie_limit"] == 1000

