"""
data/preprocess.py
===================
Preprocess the raw food dataset into the structured meals.csv format
required by the MAS pipeline.

Paths are computed relative to this file so the script works regardless
of the working directory from which it is invoked.

Mapping:
    recipe_name          -> name
    ingredients_raw      -> ingredients
    calories_kcal        -> calories
    protein_g            -> protein
    fat_g                -> fat
    carbs_g              -> carbs
    diet_types           -> diet_type
    cuisine_type         -> cuisine
    allergen_flags       -> allergens

Usage:
    python data/preprocess.py
"""

import csv
import os

# ─── Relative paths (portable — works from any working directory) ─────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(_DIR, "raw",       "food_recommendation_dataset_100k.csv")
OUTPUT_FILE = os.path.join(_DIR, "processed", "meals.csv")

FIELDNAMES = ["name", "ingredients", "calories", "protein", "fat",
              "carbs", "diet_type", "cuisine", "allergens"]


def preprocess() -> None:
    """Read the raw CSV and write the cleaned, renamed meals.csv."""
    print(f"[Preprocess] Input  -> {INPUT_FILE}")
    print(f"[Preprocess] Output -> {OUTPUT_FILE}")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Raw dataset not found: {INPUT_FILE}\n"
            "Place the 100k CSV file in data/raw/ and re-run this script."
        )

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    count = 0
    with open(INPUT_FILE, mode='r', encoding='utf-8') as infile, \
         open(OUTPUT_FILE, mode='w', encoding='utf-8', newline='') as outfile:

        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=FIELDNAMES)
        writer.writeheader()

        for row in reader:
            writer.writerow({
                "name":        row["recipe_name"],
                "ingredients": row["ingredients_raw"],
                "calories":    row["calories_kcal"],
                "protein":     row["protein_g"],
                "fat":         row["fat_g"],
                "carbs":       row["carbs_g"],
                "diet_type":   row["diet_types"],
                "cuisine":     row["cuisine_type"],
                "allergens":   row["allergen_flags"] if row.get("allergen_flags") else "none",
            })
            count += 1
            if count % 20_000 == 0:
                print(f"[Preprocess] Processed {count:,} rows...")

    print(f"[Preprocess] Complete. {count:,} rows saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    preprocess()
