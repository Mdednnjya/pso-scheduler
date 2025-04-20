# utils/integrator.py
import pandas as pd
import json
from utils.parser import parse_ingredient_line
from utils.portioning import adjust_portion
from utils.enrichment import enrich_ingredients

def process_recipe_file(recipe_json_path, tkpi_csv_path, output_json_path, target_servings=4):
    """
    Process recipe file with ingredient parsing, portion adjustment, and nutrient enrichment
    
    Args:
        recipe_json_path: Path to recipe JSON file
        tkpi_csv_path: Path to TKPI nutrition data CSV
        output_json_path: Path to save enriched recipes
        target_servings: Target number of servings to adjust portions to
    """
    # Load recipe data
    with open(recipe_json_path, 'r', encoding='utf-8') as f:
        recipes = json.load(f)
    
    enriched_recipes = []
    
    for recipe in recipes:
        # Since the ingredients are already enriched, we'll convert them
        # to the format expected by the adjust_portion function
        converted_ingredients = []
        
        for ing in recipe.get('Ingredients_Enriched', []):
            # Convert to the format expected by adjust_portion
            # Set default amount of 100g if no specific amount information
            converted = {
                "ingredient": ing.get('ingredient', ''),
                "amount": 100,  # Default to 100g per ingredient
                "unit": "gram"  # Assume gram as default unit
            }
            converted_ingredients.append(converted)
        
        # Adjust portions for target servings
        original_servings = recipe.get('Servings', 1)
        if not original_servings:
            original_servings = 1  # Default to 1 if not specified
            
        adjusted_ingredients = adjust_portion(converted_ingredients, original_servings, target_servings)
        
        # Enrich with nutrition data
        enriched_ingredients = enrich_ingredients(adjusted_ingredients, tkpi_csv_path)
        
        # Add to enriched recipes
        enriched_recipe = {
            "ID": recipe.get("ID", ""),
            "Title": recipe.get("Title", ""),
            "Original_Servings": original_servings,
            "Target_Servings": target_servings,
            "Ingredients_Parsed": recipe.get("Ingredients_Parsed", []),
            "Ingredients_Original": recipe.get("Ingredients_Enriched", []),
            "Ingredients_Adjusted": enriched_ingredients,
            "Total_Nutrition": calculate_total_nutrition(enriched_ingredients)
        }
        
        enriched_recipes.append(enriched_recipe)
    
    # Save enriched recipes
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_recipes, f, indent=2, ensure_ascii=False)
    
    print(f"✅ {len(enriched_recipes)} recipes processed and saved to '{output_json_path}'")
    return enriched_recipes

def calculate_total_nutrition(enriched_ingredients):
    """Calculate total nutrition from enriched ingredients"""
    total = {
        "calories": 0,
        "protein": 0,
        "fat": 0,
        "carbohydrates": 0,
        "dietary_fiber": 0,
        "calcium": 0
    }
    
    for ing in enriched_ingredients:
        for nutrient in total.keys():
            if ing.get(nutrient) not in [None, "", 0, 0.0]:
                try:
                    total[nutrient] += float(ing[nutrient])
                except (ValueError, TypeError):
                    continue
    
    # Round values
    for nutrient in total:
        total[nutrient] = round(total[nutrient], 1)
    
    return total