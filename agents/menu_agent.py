import pandas as pd
from config import DATA_PATH

def fetch_menu(state):
    df = pd.read_csv(DATA_PATH)

    if state["preferences"].get("diet"):
        diet = state["preferences"]["diet"]
        df = df[df["diet_type"].str.contains(diet, case=False, na=False)]

    state["candidate_meals"] = df.to_dict(orient="records")
    return state