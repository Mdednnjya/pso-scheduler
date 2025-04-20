from src.data_loader import load_data, drop_rename_fill_and_replace
from src.nutrition_enricher import process_dataset
from src.export_utils import export_to_json, export_summary_csv
import os


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

    drop_rename_fill_and_replace(input_path, output_path, columns_to_remove, rename_map)

    nutrition_path = os.path.join(data_dir, "tkpi_data.csv")

    # Load data
    recipes_df, nutrition_df = load_data(recipe_path, nutrition_path)

    # Proses data
    enriched_df = process_dataset(recipes_df, nutrition_df)

    # Export hasil
    json_output_path = os.path.join(output_dir, "enriched_recipes.json")
    csv_output_path = os.path.join(output_dir, "enriched_recipes.csv")

    export_to_json(enriched_df, json_output_path)
    export_summary_csv(enriched_df, csv_output_path)


if __name__ == "__main__":
    main()