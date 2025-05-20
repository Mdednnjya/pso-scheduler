import sys
from pathlib import Path
import os
import pandas as pd
import mlflow

sys.path.append(str(Path(__file__).parent.parent))
from src.data.data_loader import load_data, drop_rename_fill_and_replace
from src.data.nutrition_enricher import process_dataset
from src.data.export_utils import export_to_json, export_summary_csv

def calculate_coverage(enriched_df):
    """Menghitung persentase bahan yang berhasil diperkaya dengan data nutrisi"""
    total_ingredients = 0
    matched_ingredients = 0
    
    for _, row in enriched_df.iterrows():
        for ing in row['Ingredients_Enriched']:
            total_ingredients += 1
            if ing.get('calories', 0) > 0:  # Asumsi kalori > 0 berarti berhasil diperkaya
                matched_ingredients += 1
                
    return (matched_ingredients / total_ingredients) * 100 if total_ingredients > 0 else 0

def calculate_avg_ingredients(enriched_df):
    """Menghitung rata-rata jumlah bahan per resep"""
    total_recipes = len(enriched_df)
    total_ingredients = sum(len(row['Ingredients_Enriched']) for _, row in enriched_df.iterrows())
    return total_ingredients / total_recipes if total_recipes > 0 else 0

def main():
    # Definisikan path
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Data Preprocessing")

    with mlflow.start_run(run_name="Data Enrichment"):
        # Log parameters
        mlflow.log_params({
            "input_path": "data/raw/raw_nutrition_tkpi_scraped.csv",
            "nutrition_source": "TKPI 2023",
            "target_serving": 2
        })

        # Definisikan direktori
        raw_data_dir = "data/raw"
        interim_data_dir = "data/interim"
        output_dir = "output"

        # Pastikan semua direktori ada
        os.makedirs(raw_data_dir, exist_ok=True)
        os.makedirs(interim_data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Path file
        recipe_path = os.path.join(raw_data_dir, "raw_combined_dataset.csv")
        input_nutrition_path = os.path.join(raw_data_dir, "raw_nutrition_tkpi_scraped.csv")
        processed_nutrition_path = os.path.join(interim_data_dir, "processed_nutrition_tkpi.csv")

        # Preprocessing data mentah TKPI hasil scrapping
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

        # Proses data nutrisi TKPI
        drop_rename_fill_and_replace(input_nutrition_path, processed_nutrition_path, columns_to_remove, rename_map)

        # Load data
        recipes_df, nutrition_df = load_data(recipe_path, processed_nutrition_path)
        mlflow.log_metric("initial_recipes", len(recipes_df))

        # Pastikan semua nilai nutrisi dalam format yang konsisten
        numeric_columns = ['calories', 'protein', 'fat', 'carbohydrates', 'fiber', 'calcium']
        for col in numeric_columns:
            nutrition_df[col] = pd.to_numeric(nutrition_df[col], errors='coerce').fillna(0)

        # Print sample dari data nutrisi untuk debugging
        print("\nSampel 5 baris data nutrisi:")
        print(nutrition_df.head(5))

        # Proses data
        enriched_df = process_dataset(recipes_df, nutrition_df)

        # Log results
        mlflow.log_metrics({
            "final_recipes": len(enriched_df),
            "ingredient_coverage": calculate_coverage(enriched_df),
            "avg_ingredients": calculate_avg_ingredients(enriched_df)
        })

        # Save and log sample data
        enriched_df.sample(5).to_json("sample_recipes.json", orient="records")
        mlflow.log_artifact("sample_recipes.json")

        # Export hasil ke output (tetap di direktori output)
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