import pandas as pd
import numpy as np
from fuzzywuzzy import process, fuzz
import logging
from ingredient_parser import extract_ingredients, normalize_ingredient_name
from ingredient_converter import process_ingredient_weights

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('nutrition_enricher')

# Sinonim untuk membantu matching
INGREDIENT_SYNONYMS = {
    'ayam': ['chicken', 'daging ayam'],
    'bawang putih': ['garlic', 'bawang puteh'],
    'bawang merah': ['shallot', 'bawang bawang'],
    'bawang bombay': ['onion', 'bombay'],
    'cabai': ['cabe', 'chili', 'chilli'],
    'daging sapi': ['beef', 'daging', 'sapi'],
    'daging kambing': ['lamb', 'kambing', 'domba'],
    'udang': ['shrimp', 'prawn'],
    'ikan': ['fish'],
    'telur': ['egg', 'telor'],
    'tahu': ['tofu'],
    'tempe': ['tempeh'],
    'jahe': ['ginger'],
    'kunyit': ['turmeric'],
    'kencur': ['galangal'],
    'lengkuas': ['galangal'],
    'minyak': ['oil'],
    'gula': ['sugar'],
    'garam': ['salt'],
    'tepung': ['flour'],
    'santan': ['coconut milk'],
    'tomat': ['tomato'],
    'wortel': ['carrot'],
    'kentang': ['potato'],
    'kacang': ['peanut', 'bean', 'nut']
}


def expand_synonyms(ingredient_name):
    """
    Expand ingredient name with synonyms for better matching
    """
    expanded = [ingredient_name]

    # Add synonyms if available
    for key, synonyms in INGREDIENT_SYNONYMS.items():
        if key in ingredient_name:
            expanded.extend(synonyms)
            break

    # Also check if ingredient is a synonym and add the main term
    for key, synonyms in INGREDIENT_SYNONYMS.items():
        if any(synonym in ingredient_name for synonym in synonyms):
            expanded.append(key)
            break

    return expanded


def find_best_match(ingredient, nutrition_df, threshold=65):
    """
    Find the best match for an ingredient in the nutrition database

    Args:
        ingredient: Dict with ingredient info
        nutrition_df: DataFrame with nutrition data
        threshold: Minimum score to consider a match

    Returns:
        dict: Match result with nutrition data and score
    """
    if not ingredient or not ingredient.get('name'):
        return None

    # Normalize the ingredient name
    normalized_ingredient = ingredient['name']

    # Get synonyms for expanded matching
    ingredient_variations = expand_synonyms(normalized_ingredient)

    # Normalize all choices in the nutrition database
    choices = nutrition_df['ingredient'].tolist()

    # Try matching with each variation
    best_matches = []
    for variation in ingredient_variations:
        try:
            match_result = process.extractOne(
                variation,
                choices,
                scorer=fuzz.token_sort_ratio
            )
            if match_result:
                best_matches.append(match_result)
        except Exception as e:
            logger.error(f"Error in fuzzy matching: {str(e)}")

    # Find the best among all variations
    if best_matches:
        best_match = max(best_matches, key=lambda x: x[1])
        match_name, score = best_match

        if score >= threshold:
            # Get the nutrition data for the match
            match_row = nutrition_df[nutrition_df['ingredient'] == match_name].iloc[0]

            result = {
                'ingredient': match_name,
                'original_ingredient': ingredient['raw'],
                'match_score': score,
                'match_type': 'fuzzy'
            }

            # Add nutrition values
            for column in nutrition_df.columns:
                if column != 'ingredient':
                    result[column] = match_row[column]

            return result

    # If no good match found or score is below threshold
    logger.warning(f"No good match found for {normalized_ingredient}")
    return {
        'ingredient': normalized_ingredient,
        'original_ingredient': ingredient['raw'],
        'match_score': 0,
        'match_type': 'none',
        'calories': 0,
        'protein': 0,
        'fat': 0,
        'carbohydrates': 0,
        'fiber': 0,
        'calcium': 0
    }


def calculate_nutrition_for_weight(ingredient_with_match, weight_g):
    """
    Calculate nutrition values based on ingredient weight

    Args:
        ingredient_with_match: Dict with nutrition per 100g
        weight_g: Weight in grams

    Returns:
        dict: Nutrition values scaled by weight
    """
    # Calculate the scaling factor (weight in g / 100g)
    scaling_factor = weight_g / 100.0

    # Scale all numeric nutrition values
    nutrition_fields = ['calories', 'protein', 'fat', 'carbohydrates', 'fiber', 'calcium']
    result = ingredient_with_match.copy()

    for field in nutrition_fields:
        try:
            value = float(ingredient_with_match.get(field, 0))
            result[field] = value * scaling_factor
        except (ValueError, TypeError):
            result[field] = 0

    return result


def enrich_ingredients(ingredient_list, nutrition_df):
    """
    Enrich ingredients with nutrition data using fuzzy matching

    Args:
        ingredient_list: List of parsed ingredients
        nutrition_df: DataFrame with nutrition data

    Returns:
        list: Enriched ingredients with nutrition data
    """
    enriched = []

    # Process weights first
    ingredients_with_weights = process_ingredient_weights(ingredient_list)

    for ingredient in ingredients_with_weights:
        # Find the best match in the nutrition database
        match = find_best_match(ingredient, nutrition_df)

        if match:
            # Calculate nutrition based on weight
            enriched_ingredient = calculate_nutrition_for_weight(
                match,
                ingredient['weight_g']
            )

            # Add original ingredient data
            enriched_ingredient['parsed_ingredient'] = ingredient

            enriched.append(enriched_ingredient)

    logger.info(f"Enriched {len(enriched)} ingredients with nutrition data")
    return enriched


def validate_enrichment(enriched_ingredients):
    """
    Validate enrichment results and provide statistics

    Args:
        enriched_ingredients: List of enriched ingredients

    Returns:
        dict: Validation statistics
    """
    stats = {
        'total_ingredients': len(enriched_ingredients),
        'matched_ingredients': 0,
        'low_confidence_matches': 0,
        'zero_nutrition_count': 0
    }

    for ing in enriched_ingredients:
        # Check if successfully matched
        if ing.get('match_type') != 'none':
            stats['matched_ingredients'] += 1

        # Check match confidence
        if ing.get('match_score', 0) < 75:
            stats['low_confidence_matches'] += 1

        # Check if nutrition values are present
        if ing.get('calories', 0) == 0:
            stats['zero_nutrition_count'] += 1

    # Calculate match rate
    stats['match_rate'] = (stats['matched_ingredients'] / stats['total_ingredients'] * 100
                           if stats['total_ingredients'] > 0 else 0)

    return stats


def calculate_recipe_nutrition(enriched_ingredients):
    """
    Calculate total nutrition for a recipe based on enriched ingredients

    Args:
        enriched_ingredients: List of enriched ingredients

    Returns:
        dict: Total nutrition values
    """
    total_nutrition = {
        'calories': 0,
        'protein': 0,
        'fat': 0,
        'carbohydrates': 0,
        'fiber': 0,
        'calcium': 0
    }

    for ing in enriched_ingredients:
        for nutrient in total_nutrition:
            try:
                value = float(ing.get(nutrient, 0) or 0)
                total_nutrition[nutrient] += value
            except (ValueError, TypeError):
                pass

    # Round values for readability
    for nutrient in total_nutrition:
        total_nutrition[nutrient] = round(total_nutrition[nutrient], 1)

    return total_nutrition


def estimate_recipe_portions(total_nutrition, enriched_ingredients):
    """
    Estimate number of portions for a recipe

    Args:
        total_nutrition: Dict with total nutrition values
        enriched_ingredients: List of enriched ingredients

    Returns:
        int: Estimated number of portions
    """
    # Default is 4 portions
    default_portions = 4

    # Calculate based on total calories
    # Assumption: average meal is ~500-700 calories per person
    total_calories = total_nutrition.get('calories', 0)

    if total_calories <= 0:
        return default_portions

    # Check if this is likely a side dish (low calories, few ingredients)
    if total_calories < 300 and len(enriched_ingredients) < 5:
        return 2  # Side dish typically 2 portions

    # Main dishes estimation
    if total_calories < 1000:
        return 2
    elif total_calories < 2000:
        return 3
    elif total_calories < 2800:
        return 4
    else:
        return 5


def process_recipe(recipe_row, nutrition_df):
    """
    Process a single recipe row

    Args:
        recipe_row: DataFrame row with recipe data
        nutrition_df: DataFrame with nutrition data

    Returns:
        dict: Processed recipe with nutrition data
    """
    try:
        # Extract ingredients from text
        ingredients_text = recipe_row['Ingredients']
        parsed_ingredients = extract_ingredients(ingredients_text)

        # Enrich with nutrition data
        enriched_ingredients = enrich_ingredients(parsed_ingredients, nutrition_df)

        # Calculate total nutrition
        total_nutrition = calculate_recipe_nutrition(enriched_ingredients)

        # Estimate portions
        estimated_portions = estimate_recipe_portions(total_nutrition, enriched_ingredients)

        # Calculate per-portion nutrition
        per_portion_nutrition = {
            nutrient: value / estimated_portions
            for nutrient, value in total_nutrition.items()
        }

        # Validate enrichment
        validation_stats = validate_enrichment(enriched_ingredients)

        # Create processed recipe object
        processed_recipe = {
            'ID': recipe_row['ID'],
            'Title': recipe_row['Title'],
            'Ingredients_Parsed': parsed_ingredients,
            'Ingredients_Enriched': enriched_ingredients,
            'Total_Nutrition': total_nutrition,
            'Estimated_Portions': estimated_portions,
            'Per_Portion_Nutrition': per_portion_nutrition,
            'Enrichment_Stats': validation_stats
        }

        logger.info(f"Processed recipe {recipe_row['ID']}: {recipe_row['Title']}")
        return processed_recipe

    except Exception as e:
        logger.error(f"Error processing recipe {recipe_row['ID']}: {str(e)}")
        return None


def process_dataset(recipe_df, nutrition_df):
    """
    Process the entire recipe dataset

    Args:
        recipe_df: DataFrame with recipe data
        nutrition_df: DataFrame with nutrition data

    Returns:
        DataFrame: Processed recipes
    """
    # Ensure numeric values in nutrition data
    numeric_columns = ['calories', 'protein', 'fat', 'carbohydrates', 'fiber', 'calcium']
    for col in numeric_columns:
        nutrition_df[col] = pd.to_numeric(nutrition_df[col], errors='coerce').fillna(0)

    all_processed = []

    # Process each recipe
    for idx, row in recipe_df.iterrows():
        processed = process_recipe(row, nutrition_df)
        if processed:
            all_processed.append(processed)

    # Convert to DataFrame
    processed_df = pd.DataFrame(all_processed)

    logger.info(f"Processed {len(processed_df)} recipes out of {len(recipe_df)}")

    return processed_df