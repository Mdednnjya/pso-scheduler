import pandas as pd
from .ingredient_parser import extract_ingredients

def enrich_ingredients(ingredient_list, nutrition_df):
    """Enrich ingredients with nutrition data."""
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

def process_dataset(recipe_df, nutrition_df):
    """Process the entire recipe dataset with nutrition information."""
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