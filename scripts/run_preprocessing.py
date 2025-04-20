import sys
from pathlib import Path
import os
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from src.data.data_loader import load_data, drop_rename_fill_and_replace
from src.data.nutrition_enricher import process_dataset
from src.data.export_utils import export_to_json, export_summary_csv


def main():
    # Definisikan path
    data_dir = "data"
    output_dir = "output"

    # Buat direktori output jika belum ada
    os.makedirs(output_dir, exist_ok=True)

    recipe_path = os.path.join(data_dir, "combined_dataset.csv")

    # Preprocessing data mentah TKPI hasil scrapping
    input_path = "data/scrap_tkpi.csv"
    output_path = "data/tkpi_data.csv"
    columns_to_remove = [0, 1, 2, -1, -2]
    rename_map = {
        "nama bahan makanan": "ingredient",
        "calori": "calories",
        "protein": "protein",
        "fat": "fat",
        "carbohydrate": "carbohydrates",
        "fiber": "fiber",
        "calsium": "calcium"
    }

    # Periksa apakah fuzzywuzzy sudah terinstal
    try:
        import fuzzywuzzy
    except ImportError:
        print("Installing fuzzywuzzy package...")
        import pip
        pip.main(['install', 'fuzzywuzzy'])
        pip.main(['install', 'python-Levenshtein'])  # Optional but improves performance

    drop_rename_fill_and_replace(input_path, output_path, columns_to_remove, rename_map)

    nutrition_path = os.path.join(data_dir, "tkpi_data.csv")

    # Load data
    recipes_df, nutrition_df = load_data(recipe_path, nutrition_path)

    # Pastikan semua nilai nutrisi dalam format yang konsisten
    numeric_columns = ['calories', 'protein', 'fat', 'carbohydrates', 'fiber', 'calcium']
    for col in numeric_columns:
        nutrition_df[col] = pd.to_numeric(nutrition_df[col], errors='coerce').fillna(0)

    # Print sample dari data nutrisi untuk debugging
    print("\nSampel 5 baris data nutrisi:")
    print(nutrition_df.head(5))

    # Proses data
    enriched_df = process_dataset(recipes_df, nutrition_df)

    # Export hasil
    json_output_path = os.path.join(output_dir, "enriched_recipes.json")
    csv_output_path = os.path.join(output_dir, "enriched_recipes.csv")

    export_to_json(enriched_df, json_output_path)
    export_summary_csv(enriched_df, csv_output_path)

    # Tampilkan beberapa statistik
    print("\nStatistik Hasil Enrichment:")
    count_with_nutrition = 0
    total_recipes = len(enriched_df)

    for _, row in enriched_df.iterrows():
        total_calories = sum(float(ing.get('calories', 0) or 0) for ing in row['Ingredients_Enriched'])
        if total_calories > 0:
            count_with_nutrition += 1

    print(f"Total resep: {total_recipes}")
    print(f"Resep dengan data nutrisi: {count_with_nutrition} ({count_with_nutrition / total_recipes * 100:.1f}%)")


if __name__ == "__main__":
    main()