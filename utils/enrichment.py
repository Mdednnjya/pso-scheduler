import pandas as pd
from fuzzywuzzy import process

def normalize_ingredient_name(name):
    """Normalize ingredient name for better matching."""
    return name.lower().strip()

def find_best_match(ingredient, nutrition_df, threshold=65):
    """Find the best match for an ingredient using fuzzy matching."""
    if not ingredient:
        return None
    
    normalized_ingredient = normalize_ingredient_name(ingredient)
    if not normalized_ingredient:
        return None
    
    # Create a list of normalized ingredient names from the nutrition database
    choices = nutrition_df['ingredient'].apply(normalize_ingredient_name).tolist()
    
    # Create a mapping from normalized names to original indices
    choice_map = {normalize_ingredient_name(choice): idx for idx, choice in 
                enumerate(nutrition_df['ingredient'].tolist())}
    
    # Find the best match using fuzzy matching
    try:
        best_match, score = process.extractOne(normalized_ingredient, choices)
        
        if score >= threshold:
            original_idx = choice_map[best_match]
            return nutrition_df.iloc[original_idx]
    except:
        pass
    
    return None

def enrich_ingredients(ingredients, tkpi_csv_path):
    """Enrich ingredients with nutrition data."""
    # Load the nutrition data
    nutrition_df = pd.read_csv(tkpi_csv_path)
    
    enriched = []
    for item in ingredients:
        name = item['ingredient']
        
        # Try exact matching first
        match = nutrition_df[nutrition_df['ingredient'].str.lower() == name.lower()]
        
        if not match.empty:
            data = match.iloc[0]
            # Calculate nutrition based on adjusted amount
            adjusted_amount = item.get('adjusted_amount', item['amount'])
            
            enriched.append({
                **item,
                "calories": round(float(data["calories"]) * adjusted_amount / 100, 2),
                "protein": round(float(data["protein"]) * adjusted_amount / 100, 2),
                "fat": round(float(data["fat"]) * adjusted_amount / 100, 2),
                "carbohydrates": round(float(data["carbohydrates"]) * adjusted_amount / 100, 2),
                "dietary_fiber": round(float(data.get("fiber", 0)) * adjusted_amount / 100, 2),
                "calcium": round(float(data.get("calcium", 0)) * adjusted_amount / 100, 2),
            })
        else:
            # Try fuzzy matching
            match = find_best_match(name, nutrition_df)
            if match is not None:
                # Calculate nutrition based on adjusted amount
                adjusted_amount = item.get('adjusted_amount', item['amount'])
                
                enriched.append({
                    **item,
                    "calories": round(float(match["calories"]) * adjusted_amount / 100, 2),
                    "protein": round(float(match["protein"]) * adjusted_amount / 100, 2),
                    "fat": round(float(match["fat"]) * adjusted_amount / 100, 2),
                    "carbohydrates": round(float(match["carbohydrates"]) * adjusted_amount / 100, 2),
                    "dietary_fiber": round(float(match.get("fiber", 0)) * adjusted_amount / 100, 2),
                    "calcium": round(float(match.get("calcium", 0)) * adjusted_amount / 100, 2),
                    "match_type": "fuzzy"
                })
            else:
                # Ingredient not found, include it with empty nutrition values
                enriched.append({
                    **item,
                    "calories": 0,
                    "protein": 0,
                    "fat": 0,
                    "carbohydrates": 0,
                    "dietary_fiber": 0,
                    "calcium": 0,
                    "match_type": "none"
                })
    
    return enriched