import pandas as pd
import json
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('export_utils')


def export_to_json(df, output_path):
    """
    Export DataFrame to JSON file

    Args:
        df: DataFrame to export
        output_path: Path to output JSON file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert DataFrame to list of dictionaries
    records = df.to_dict(orient='records')

    # Write to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    logger.info(f"Data successfully exported to '{output_path}'")


def export_summary_csv(enriched_df, output_path):
    """
    Export nutrition summary to CSV file

    Args:
        enriched_df: DataFrame with enriched recipe data
        output_path: Path to output CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Extract summary data
    rows = []

    for _, row in enriched_df.iterrows():
        # Get recipe info
        recipe_id = row['ID']
        title = row['Title']

        # Get total nutrition
        total_nutrition = row['Total_Nutrition']

        # Get match rate information
        enrichment_stats = row.get('Enrichment_Stats', {})
        match_rate = enrichment_stats.get('match_rate', 0)

        # Create summary row
        summary_row = {
            'ID': recipe_id,
            'Title': title,
            'Calories': round(total_nutrition.get('calories', 0), 1),
            'Protein': round(total_nutrition.get('protein', 0), 1),
            'Fat': round(total_nutrition.get('fat', 0), 1),
            'Carbohydrates': round(total_nutrition.get('carbohydrates', 0), 1),
            'Fiber': round(total_nutrition.get('fiber', 0), 1),
            'Calcium': round(total_nutrition.get('calcium', 0), 1),
            'Estimated_Portions': row.get('Estimated_Portions', 4),
            'Match_Rate': f"{match_rate:.1f}%"
        }

        rows.append(summary_row)

    # Create DataFrame and export to CSV
    df_export = pd.DataFrame(rows)
    df_export.to_csv(output_path, index=False)

    logger.info(f"Nutrition summary exported to '{output_path}'")


def export_validation_report(enriched_df, output_path):
    """
    Export validation report with matching statistics

    Args:
        enriched_df: DataFrame with enriched recipe data
        output_path: Path to output JSON file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Extract validation data
    validation_report = {
        'overall_stats': {
            'total_recipes': len(enriched_df),
            'total_ingredients': 0,
            'matched_ingredients': 0,
            'low_confidence_matches': 0,
            'zero_nutrition': 0
        },
        'recipes': []
    }

    # Process each recipe
    for _, row in enriched_df.iterrows():
        recipe_stats = row.get('Enrichment_Stats', {})

        # Update overall stats
        validation_report['overall_stats']['total_ingredients'] += recipe_stats.get('total_ingredients', 0)
        validation_report['overall_stats']['matched_ingredients'] += recipe_stats.get('matched_ingredients', 0)
        validation_report['overall_stats']['low_confidence_matches'] += recipe_stats.get('low_confidence_matches', 0)
        validation_report['overall_stats']['zero_nutrition'] += recipe_stats.get('zero_nutrition_count', 0)

        # Add recipe-specific stats
        recipe_report = {
            'ID': row['ID'],
            'Title': row['Title'],
            'match_rate': recipe_stats.get('match_rate', 0),
            'total_ingredients': recipe_stats.get('total_ingredients', 0),
            'matched_ingredients': recipe_stats.get('matched_ingredients', 0),
            'low_confidence_matches': recipe_stats.get('low_confidence_matches', 0)
        }

        validation_report['recipes'].append(recipe_report)

    # Calculate overall match rate
    total_ing = validation_report['overall_stats']['total_ingredients']
    matched_ing = validation_report['overall_stats']['matched_ingredients']

    if total_ing > 0:
        validation_report['overall_stats']['overall_match_rate'] = (matched_ing / total_ing) * 100
    else:
        validation_report['overall_stats']['overall_match_rate'] = 0

    # Write to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(validation_report, f, indent=2, ensure_ascii=False)

    logger.info(f"Validation report exported to '{output_path}'")