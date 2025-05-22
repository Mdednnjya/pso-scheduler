import sys
from pathlib import Path
import os
import pandas as pd
import mlflow
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import load_data, drop_rename_fill_and_replace
from src.data.nutrition_enricher import process_dataset
from src.data.export_utils import export_to_json, export_summary_csv, export_validation_report

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('preprocessing')


def calculate_coverage(enriched_df):
    """Calculate percentage of ingredients successfully enriched with nutrition data"""
    total_matches = 0
    total_ingredients = 0

    for _, row in enriched_df.iterrows():
        stats = row.get('Enrichment_Stats', {})
        total_ingredients += stats.get('total_ingredients', 0)
        total_matches += stats.get('matched_ingredients', 0)

    return (total_matches / total_ingredients) * 100 if total_ingredients > 0 else 0


def calculate_avg_ingredients(enriched_df):
    """Calculate average number of ingredients per recipe"""
    total_recipes = len(enriched_df)
    total_ingredients = sum(row.get('Enrichment_Stats', {}).get('total_ingredients', 0)
                            for _, row in enriched_df.iterrows())

    return total_ingredients / total_recipes if total_recipes > 0 else 0


def print_summary(enriched_df):
    """Print summary statistics for the enriched dataset"""
    logger.info("\n" + "=" * 50)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("=" * 50)

    # Basic counts
    total_recipes = len(enriched_df)
    avg_ingredients = calculate_avg_ingredients(enriched_df)

    logger.info(f"Total recipes processed: {total_recipes}")
    logger.info(f"Average ingredients per recipe: {avg_ingredients:.1f}")

    # Enrichment coverage
    coverage = calculate_coverage(enriched_df)
    logger.info(f"Overall ingredient match rate: {coverage:.1f}%")

    # Nutrition statistics
    if 'Total_Nutrition' in enriched_df.columns:
        avg_calories = enriched_df['Total_Nutrition'].apply(lambda x: x.get('calories', 0)).mean()
        avg_protein = enriched_df['Total_Nutrition'].apply(lambda x: x.get('protein', 0)).mean()

        logger.info(f"Average calories per recipe: {avg_calories:.1f}")
        logger.info(f"Average protein per recipe: {avg_protein:.1f}g")

    # Portion statistics
    if 'Estimated_Portions' in enriched_df.columns:
        portion_counts = enriched_df['Estimated_Portions'].value_counts()
        logger.info(f"Portion distribution: {portion_counts.to_dict()}")

    logger.info("=" * 50)


def main():
    # Define paths
    logger.info("Starting preprocessing pipeline")

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Data Preprocessing")

    with mlflow.start_run(run_name="Advanced Enrichment"):
        # Define directories
        raw_data_dir = "data/raw"
        interim_data_dir = "data/interim"
        output_dir = "output"

        # Create directories if they don't exist
        for directory in [raw_data_dir, interim_data_dir, output_dir]:
            os.makedirs(directory, exist_ok=True)

        # Log parameters
        mlflow.log_params({
            "raw_nutrition_path": f"{raw_data_dir}/raw_nutrition_tkpi_scraped.csv",
            "supplementary_nutrition_path": f"{raw_data_dir}/raw_nutrition_supplement.csv",
            "raw_recipes_path": f"{raw_data_dir}/raw_combined_dataset.csv",
            "nutrition_source": "TKPI 2023 + Supplements",
            "matching_algorithm": "Enhanced Fuzzy with synonyms",
            "matching_threshold": 65
        })

        # Define paths
        recipe_path = os.path.join(raw_data_dir, "raw_combined_dataset.csv")
        input_nutrition_path = os.path.join(raw_data_dir, "raw_nutrition_tkpi_scraped.csv")
        input_supplement_path = os.path.join(raw_data_dir, "raw_nutrition_supplement.csv")
        processed_nutrition_path = os.path.join(interim_data_dir, "processed_nutrition_tkpi.csv")

        # Process TKPI data
        logger.info(f"Processing nutrition data from {input_nutrition_path}")

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

        # Ensure required packages are installed
        try:
            import fuzzywuzzy
        except ImportError:
            logger.warning("Installing fuzzywuzzy package...")
            import pip
            pip.main(['install', 'fuzzywuzzy'])
            pip.main(['install', 'python-Levenshtein'])  # Optional but improves performance
            import fuzzywuzzy

        # Process nutrition data
        drop_rename_fill_and_replace(input_nutrition_path, processed_nutrition_path, columns_to_remove, rename_map)

        # Load data
        logger.info(f"Loading recipe data from {recipe_path}")
        recipes_df, nutrition_df = load_data(recipe_path, processed_nutrition_path)

        # Load supplementary nutrition data if available
        supplement_df = None
        if os.path.exists(input_supplement_path):
            logger.info(f"Loading supplementary nutrition data from {input_supplement_path}")
            supplement_df = pd.read_csv(input_supplement_path)
            logger.info(f"Loaded {len(supplement_df)} supplementary ingredient entries")
        else:
            logger.warning(f"Supplementary nutrition file not found: {input_supplement_path}")

        mlflow.log_metric("initial_recipes", len(recipes_df))

        # Ensure numeric columns in nutrition data
        logger.info("Normalizing nutrition data types")
        numeric_columns = ['calories', 'protein', 'fat', 'carbohydrates', 'fiber', 'calcium']
        for col in numeric_columns:
            nutrition_df[col] = pd.to_numeric(nutrition_df[col], errors='coerce').fillna(0)
            if supplement_df is not None and col in supplement_df.columns:
                supplement_df[col] = pd.to_numeric(supplement_df[col], errors='coerce').fillna(0)

        # Process recipes
        logger.info("Processing recipes with enhanced nutrition enrichment")
        enriched_df = process_dataset(recipes_df, nutrition_df, supplement_df)

        # Log metrics
        coverage = calculate_coverage(enriched_df)
        avg_ingredients = calculate_avg_ingredients(enriched_df)

        mlflow.log_metrics({
            "final_recipes": len(enriched_df),
            "ingredient_coverage": coverage,
            "avg_ingredients": avg_ingredients
        })

        # Save sample for MLflow
        sample_path = "sample_recipes.json"
        enriched_df.sample(min(5, len(enriched_df))).to_json(sample_path, orient="records")
        mlflow.log_artifact(sample_path)
        os.remove(sample_path)  # Clean up

        # Export results
        logger.info("Exporting enriched data")
        json_output_path = os.path.join(output_dir, "enriched_recipes.json")
        csv_output_path = os.path.join(output_dir, "enriched_recipes.csv")
        validation_output_path = os.path.join(output_dir, "enrichment_validation.json")

        export_to_json(enriched_df, json_output_path)
        export_summary_csv(enriched_df, csv_output_path)
        export_validation_report(enriched_df, validation_output_path)

        # Print summary
        print_summary(enriched_df)

        logger.info("Preprocessing pipeline completed successfully!")


if __name__ == "__main__":
    main()