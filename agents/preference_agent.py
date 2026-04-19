import json
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import JsonOutputParser
from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from tools.logger import log_agent_step

# Define the expected JSON output format
PREFERENCE_PROMPT = """
You are an expert dietary assistant. Your task is to extract dietary preferences from the user's input and format it as a valid JSON object.

Extract the following:
- diet: The diet type (e.g., "vegan", "vegetarian", "keto"). If none, use null.
- calorie_limit: An integer representing the maximum calories. If not specified, use null.
- exclude: A list of strings for avoiding ingredients/allergies (e.g., ["nuts", "dairy"]). If none, use an empty list [].
- cuisine: The type of cuisine (e.g., "Italian", "Mexican"). If none, use null.

User Input: "I want vegan food under 500 calories without nuts"
Expected output:
{{
    "diet": "vegan",
    "calorie_limit": 500,
    "exclude": ["nuts"],
    "cuisine": null
}}

User Input: "{user_input}"
Expected output:
"""

def analyze_preferences(state, user_input):
    print(f"Agent 1: Analyzing preferences for input -> '{user_input}'")
    
    try:
        # Set up the LLM with Ollama
        llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, format="json", temperature=0)
        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template=PREFERENCE_PROMPT,
            input_variables=["user_input"]
        )
        
        chain = prompt | llm | parser
        
        # Invoke the chain
        preferences = chain.invoke({"user_input": user_input})
        
        # Ensure fallback defaults if LLM misses keys
        extracted_data = {
            "diet": preferences.get("diet", None),
            "calorie_limit": preferences.get("calorie_limit", None),
            "exclude": preferences.get("exclude", []),
            "cuisine": preferences.get("cuisine", None)
        }
        print("Agent 1: Successfully extracted preferences via Ollama.")
        
    except Exception as e:
        print(f"Warning: Extractor failed (Ollama may be off). Using rules-based fallback. Error: {e}")
        # Basic heuristic Fallback if Ollama is disconnected
        user_input_low = user_input.lower()
        extracted_data = {
            "diet": "vegan" if "vegan" in user_input_low else ("vegetarian" if "veg" in user_input_low else None),
            "calorie_limit": 500 if "500" in user_input_low else (1000 if "low" in user_input_low else None),
            "exclude": ["nuts"] if "nut" in user_input_low else ([] if "dairy" not in user_input_low else ["dairy"]),
            "cuisine": "Italian" if "italian" in user_input_low else ("Mexican" if "mexican" in user_input_low else None)
        }

    # Update state
    state["preferences"] = extracted_data
    
    # Log the step
    log_agent_step(
        state=state,
        agent_name="PreferenceAnalyzerAgent",
        input_data={"user_input": user_input},
        output_data=extracted_data
    )
    
    return state