import pandas as pd
import re

# --- STEP 1: LOAD DATA ---
def load_data(recipe_path, nutrition_path):
    recipes_df = pd.read_csv(recipe_path)
    nutrition_df = pd.read_csv(nutrition_path)
    return recipes_df, nutrition_df

# --- STEP 2: PARSE INGREDIENT LINE ---
def parse_ingredient_line(line):
    # Menghapus satuan seperti "kg", "gram", dll. dan hanya ambil nama bahan
    line = line.lower()
    # Hilangkan angka dan satuan
    line = re.sub(r'[\d/.,]+', '', line)
    line = re.sub(r'\b(kg|gram|gr|sdm|sdt|butir|buah|siung|ikat|batang|lembar|mangkok|sendok|ml|cc)\b', '', line)
    # Hilangkan karakter non-alfabet
    line = re.sub(r'[^a-zA-Z\s]', '', line)
    # Normalisasi dan strip whitespace
    return line.strip()

# --- STEP 3: EXTRACT INGREDIENTS FROM RECIPE STRING ---
def extract_ingredients(ingredient_text):
    lines = re.split(r'--|\n|-', ingredient_text)
    parsed = [parse_ingredient_line(line) for line in lines if line.strip()]
    return list(set(parsed))  # Unikkan

# --- STEP 4: ENRICH WITH NUTRITIONAL DATA ---
def enrich_ingredients(ingredient_list, nutrition_df):
    enriched = []
    for item in ingredient_list:
        match = nutrition_df[nutrition_df['ingredient'].str.lower() == item.lower()]
        if not match.empty:
            row = match.iloc[0].to_dict()
        else:
            row = {
                'ingredient': item,
                'calories': None,
                'protein': None,
                'fat': None,
                'carbohydrates': None,
                'fiber': None,
                'calcium': None
            }
        enriched.append(row)
    return enriched

# --- STEP 5: APPLY TO WHOLE DATASET ---
def process_dataset(recipe_df, nutrition_df):
    all_enriched = []

    for idx, row in recipe_df.iterrows():
        ingredient_text = row['Ingredients']
        ingredients = extract_ingredients(ingredient_text)
        enriched = enrich_ingredients(ingredients, nutrition_df)
        all_enriched.append({
            'ID': row['ID'],
            'Title': row['Title'],
            'Ingredients_Parsed': ingredients,
            'Ingredients_Enriched': enriched
        })

    return pd.DataFrame(all_enriched)

def export_summary_csv(enriched_df, output_path):
    rows = []
    for _, row in enriched_df.iterrows():
        recipe_id = row['ID']
        title = row['Title']
        total = {
            'calories': 0.0,
            'protein': 0.0,
            'fat': 0.0,
            'carbohydrates': 0.0,
            'fiber': 0.0,
            'calcium': 0.0
        }
        for ing in row['Ingredients_Enriched']:
            # Hanya jumlahkan jika data gizi tersedia
            for key in total:
                if ing.get(key) not in [None, ""]:
                    try:
                        total[key] += float(ing[key])
                    except:
                        continue
        rows.append({
            'ID': recipe_id,
            'Title': title,
            'Calories': round(total['calories'], 2),
            'Protein': round(total['protein'], 2),
            'Fat': round(total['fat'], 2),
            'Carbohydrates': round(total['carbohydrates'], 2),
            'Fiber': round(total['fiber'], 2),
            'Calcium': round(total['calcium'], 2),
        })

    df_export = pd.DataFrame(rows)
    df_export.to_csv(output_path, index=False)
    print(f"✅ Ringkasan gizi per resep berhasil disimpan ke '{output_path}'")



# --- MAIN EXECUTION ---
if __name__ == "__main__":
    recipe_path = "combined_dataset.csv"
    nutrition_path = "tkpi_data.csv"

    recipes_df, nutrition_df = load_data(recipe_path, nutrition_path)
    enriched_df = process_dataset(recipes_df, nutrition_df)

    # Save hasilnya ke file
    enriched_df.to_json("enriched_recipes.json", orient='records', indent=2, force_ascii=False)
    # Ekspor juga ke CSV
    export_summary_csv(enriched_df, "enriched_recipes.csv")

    print("✅ Data berhasil diproses dan disimpan ke 'enriched_recipes.json'")
