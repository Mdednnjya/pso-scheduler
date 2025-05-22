import sys
from pathlib import Path
import os
import pandas as pd
import mlflow
import logging
import json
import argparse

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


def extract_unmatched_ingredients(enriched_df):
    """
    Extract and analyze ingredients with match_score = 0 for troubleshooting

    Args:
        enriched_df: DataFrame with enriched recipe data

    Returns:
        dict: Analysis of unmatched ingredients
    """
    unmatched_analysis = {
        'summary': {
            'total_recipes': len(enriched_df),
            'recipes_with_unmatched': 0,
            'total_unmatched_ingredients': 0,
            'unique_unmatched_ingredients': set()
        },
        'unmatched_by_recipe': [],
        'unmatched_ingredients_list': [],
        'common_patterns': {}
    }

    for _, row in enriched_df.iterrows():
        recipe_unmatched = []

        # Check each enriched ingredient
        for ingredient in row.get('Ingredients_Enriched', []):
            if ingredient.get('match_score', 100) == 0:
                unmatched_ingredient = {
                    'original_ingredient': ingredient.get('original_ingredient', ''),
                    'parsed_name': ingredient.get('ingredient', ''),
                    'parsed_data': ingredient.get('parsed_ingredient', {}),
                    'recipe_id': row['ID'],
                    'recipe_title': row['Title']
                }

                recipe_unmatched.append(unmatched_ingredient)
                unmatched_analysis['unmatched_ingredients_list'].append(unmatched_ingredient)
                unmatched_analysis['summary']['unique_unmatched_ingredients'].add(
                    ingredient.get('ingredient', '')
                )

        if recipe_unmatched:
            unmatched_analysis['summary']['recipes_with_unmatched'] += 1
            unmatched_analysis['unmatched_by_recipe'].append({
                'recipe_id': row['ID'],
                'recipe_title': row['Title'],
                'unmatched_count': len(recipe_unmatched),
                'unmatched_ingredients': recipe_unmatched
            })

    # Update summary stats
    unmatched_analysis['summary']['total_unmatched_ingredients'] = len(
        unmatched_analysis['unmatched_ingredients_list']
    )
    unmatched_analysis['summary']['unique_unmatched_count'] = len(
        unmatched_analysis['summary']['unique_unmatched_ingredients']
    )

    # Convert set to list for JSON serialization
    unmatched_analysis['summary']['unique_unmatched_ingredients'] = list(
        unmatched_analysis['summary']['unique_unmatched_ingredients']
    )

    # Analyze common patterns in unmatched ingredients
    pattern_analysis = {}

    for ingredient in unmatched_analysis['unmatched_ingredients_list']:
        parsed_name = ingredient['parsed_name'].lower()

        # Check for common patterns
        if 'instruction:' in parsed_name:
            pattern_analysis['instructions'] = pattern_analysis.get('instructions', 0) + 1
        elif any(word in parsed_name for word in ['wajan', 'panci', 'kompor', 'alat']):
            pattern_analysis['cooking_tools'] = pattern_analysis.get('cooking_tools', 0) + 1
        elif any(word in parsed_name for word in ['bahan', 'adonan', 'isian']):
            pattern_analysis['generic_terms'] = pattern_analysis.get('generic_terms', 0) + 1
        elif parsed_name == '' or parsed_name.isspace():
            pattern_analysis['empty_names'] = pattern_analysis.get('empty_names', 0) + 1
        else:
            pattern_analysis['genuine_ingredients'] = pattern_analysis.get('genuine_ingredients', 0) + 1

    unmatched_analysis['common_patterns'] = pattern_analysis

    return unmatched_analysis


def save_unmatched_analysis(unmatched_analysis, output_dir):
    """
    Save unmatched ingredients analysis to files for easy troubleshooting

    Args:
        unmatched_analysis: Dict with unmatched ingredients analysis
        output_dir: Directory to save analysis files
    """
    # Save full analysis
    analysis_file = os.path.join(output_dir, "unmatched_ingredients_analysis.json")
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(unmatched_analysis, f, indent=2, ensure_ascii=False)

    # Save simple list for easy copy-paste
    simple_list = []
    for ingredient in unmatched_analysis['unmatched_ingredients_list']:
        simple_list.append({
            'original': ingredient['original_ingredient'],
            'parsed': ingredient['parsed_name'],
            'recipe': f"ID {ingredient['recipe_id']}: {ingredient['recipe_title']}"
        })

    simple_file = os.path.join(output_dir, "unmatched_ingredients_simple.json")
    with open(simple_file, 'w', encoding='utf-8') as f:
        json.dump(simple_list, f, indent=2, ensure_ascii=False)

    # Save unique ingredients only (for synonym development)
    unique_file = os.path.join(output_dir, "unique_unmatched_ingredients.json")
    with open(unique_file, 'w', encoding='utf-8') as f:
        json.dump({
            'count': len(unmatched_analysis['summary']['unique_unmatched_ingredients']),
            'ingredients': sorted(unmatched_analysis['summary']['unique_unmatched_ingredients'])
        }, f, indent=2, ensure_ascii=False)

    # Save pattern analysis
    pattern_file = os.path.join(output_dir, "unmatched_patterns_analysis.json")
    with open(pattern_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': unmatched_analysis['summary'],
            'patterns': unmatched_analysis['common_patterns'],
            'recommendations': {
                'instructions': 'These should be filtered out during parsing',
                'cooking_tools': 'These should be filtered out during parsing',
                'generic_terms': 'These need better normalization',
                'genuine_ingredients': 'These need new synonyms or nutrition entries'
            }
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"Unmatched ingredients analysis saved to:")
    logger.info(f"  - Full analysis: {analysis_file}")
    logger.info(f"  - Simple list: {simple_file}")
    logger.info(f"  - Unique ingredients: {unique_file}")
    logger.info(f"  - Pattern analysis: {pattern_file}")


def print_unmatched_summary(unmatched_analysis):
    """Print summary of unmatched ingredients for immediate feedback"""
    summary = unmatched_analysis['summary']
    patterns = unmatched_analysis['common_patterns']

    logger.info("\n" + "=" * 60)
    logger.info("UNMATCHED INGREDIENTS ANALYSIS")
    logger.info("=" * 60)

    logger.info(f"Total recipes analyzed: {summary['total_recipes']}")
    logger.info(f"Recipes with unmatched ingredients: {summary['recipes_with_unmatched']}")
    logger.info(f"Total unmatched ingredient instances: {summary['total_unmatched_ingredients']}")
    logger.info(f"Unique unmatched ingredients: {summary['unique_unmatched_count']}")

    if patterns:
        logger.info("\nBreakdown by pattern:")
        for pattern, count in patterns.items():
            percentage = (count / summary['total_unmatched_ingredients']) * 100
            logger.info(f"  - {pattern.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

    # Show some examples of genuine ingredients that need attention
    genuine_ingredients = []
    for ingredient in unmatched_analysis['unmatched_ingredients_list']:
        parsed_name = ingredient['parsed_name'].lower()
        if (not parsed_name.startswith('instruction:') and
                not any(word in parsed_name for word in ['wajan', 'panci', 'kompor', 'alat', 'bahan', 'adonan']) and
                parsed_name.strip() != ''):
            genuine_ingredients.append(ingredient['parsed_name'])

    if genuine_ingredients:
        unique_genuine = list(set(genuine_ingredients))[:10]  # Show first 10 unique
        logger.info(f"\nSample genuine ingredients needing attention:")
        for ing in unique_genuine:
            logger.info(f"  - '{ing}'")

        if len(unique_genuine) == 10:
            logger.info(f"  ... and {len(set(genuine_ingredients)) - 10} more")

    logger.info("=" * 60)


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
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Recipe preprocessing with nutrition enrichment')
    parser.add_argument('--debug-unmatched', action='store_true',
                        help='Generate unmatched ingredients analysis files for troubleshooting')

    args = parser.parse_args()

    # Define paths
    logger.info("Starting preprocessing pipeline")

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Data Preprocessing")

    with mlflow.start_run(run_name="Advanced Enrichment with Conditional Analysis"):
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
            "matching_threshold": 70,
            "debug_unmatched": args.debug_unmatched
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

        # **ARGPARSE-CONTROLLED: Analyze unmatched ingredients only if requested**
        if args.debug_unmatched:
            logger.info("🔍 Debug flag detected: Analyzing unmatched ingredients for troubleshooting...")
            unmatched_analysis = extract_unmatched_ingredients(enriched_df)

            # Save unmatched analysis files
            save_unmatched_analysis(unmatched_analysis, output_dir)

            # Print unmatched summary
            print_unmatched_summary(unmatched_analysis)

            # Log additional metrics for unmatched analysis
            mlflow.log_metrics({
                "unmatched_ingredients": unmatched_analysis['summary']['total_unmatched_ingredients'],
                "unmatched_recipes": unmatched_analysis['summary']['recipes_with_unmatched'],
                "unique_unmatched": unmatched_analysis['summary']['unique_unmatched_count']
            })

            # Log unmatched analysis to MLflow
            unmatched_analysis_path = os.path.join(output_dir, "unmatched_ingredients_analysis.json")
            mlflow.log_artifact(unmatched_analysis_path, "troubleshooting")

            logger.info(f"\n📁 Unmatched analysis files generated:")
            logger.info(f"  - output/unmatched_ingredients_simple.json (for copy-paste)")
            logger.info(f"  - output/unique_unmatched_ingredients.json (for synonym development)")
            logger.info(f"  - output/unmatched_patterns_analysis.json (for pattern analysis)")
        else:
            logger.info("Skipping unmatched ingredients analysis (use --debug-unmatched to enable)")

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

        if not args.debug_unmatched:
            logger.info(f"\n💡 To troubleshoot unmatched ingredients, run:")
            logger.info(f"   python scripts/run_preprocessing.py --debug-unmatched")


if __name__ == "__main__":
    main()