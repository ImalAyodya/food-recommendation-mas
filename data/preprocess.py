import csv
import os

input_file = r"d:\Y4S1\CTSE\Assignment 2\food-recommendation-mas\data\raw\food_recommendation_dataset_100k.csv"
output_file = r"d:\Y4S1\CTSE\Assignment 2\food-recommendation-mas\data\processed\meals.csv"

# Mapping from source columns to requested columns:
# recipe_name -> name
# ingredients_raw -> ingredients
# calories_kcal -> calories
# protein_g -> protein
# fat_g -> fat
# carbs_g -> carbs
# diet_types -> diet_type
# cuisine_type -> cuisine
# allergen_flags -> allergens

def preprocess():
    print("Starting preprocessing...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(input_file, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        
        with open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
            fieldnames = ['name', 'ingredients', 'calories', 'protein', 'fat', 'carbs', 'diet_type', 'cuisine', 'allergens']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            count = 0
            for row in reader:
                writer.writerow({
                    'name': row['recipe_name'],
                    'ingredients': row['ingredients_raw'],
                    'calories': row['calories_kcal'],
                    'protein': row['protein_g'],
                    'fat': row['fat_g'],
                    'carbs': row['carbs_g'],
                    'diet_type': row['diet_types'],
                    'cuisine': row['cuisine_type'],
                    'allergens': row['allergen_flags'] if row['allergen_flags'] else 'none'
                })
                count += 1
                if count % 20000 == 0:
                    print(f"Processed {count} rows...")
    print(f"Preprocessing complete. Total {count} rows saved to {output_file}")

if __name__ == "__main__":
    preprocess()
