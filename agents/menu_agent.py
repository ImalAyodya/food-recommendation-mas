import pandas as pd
from config import DATA_PATH

def fetch_menu(state):
    df = pd.read_csv(DATA_PATH)

    if state["preferences"]["diet"]:
        df = df[df["diet_type"] == state["preferences"]["diet"]]

    state["candidate_meals"] = df.to_dict(orient="records")
    return state