import pandas as pd
from fuzzywuzzy import process
from .ingredient_parser import extract_ingredients


def normalize_ingredient_name(name):
    """Normalize ingredient name for better matching."""
    # Hapus kata-kata umum yang tidak relevan
    words_to_remove = ['segar', 'rebus', 'mentah', 'masakan', 'goreng', 'kering']
    normalized = name.lower()

    for word in words_to_remove:
        normalized = normalized.replace(word, '')

    # Hapus karakter non-alfabet dan trim whitespace
    normalized = ''.join([c for c in normalized if c.isalnum() or c.isspace()])
    normalized = ' '.join(normalized.split())

    return normalized.strip()


def find_best_match(ingredient, nutrition_df, threshold=65):
    """Find the best match for an ingredient using fuzzy matching."""
    if not ingredient or ingredient.isspace():
        return None

    # Normalisasi nama bahan untuk meningkatkan matching
    normalized_ingredient = normalize_ingredient_name(ingredient)
    if not normalized_ingredient:
        return None

    # Normalisasi semua pilihan dalam database nutrisi
    normalized_choices = nutrition_df['ingredient'].apply(normalize_ingredient_name).tolist()
    choice_map = {normalize_ingredient_name(choice): idx for idx, choice in
                  enumerate(nutrition_df['ingredient'].tolist())}

    # Cari kecocokan terbaik
    try:
        best_match, score = process.extractOne(normalized_ingredient, normalized_choices)

        if score >= threshold:
            original_idx = choice_map[best_match]
            return nutrition_df.iloc[original_idx].to_dict()
    except:
        pass

    return None


def enrich_ingredients(ingredient_list, nutrition_df):
    """Enrich ingredients with nutrition data using fuzzy matching."""
    enriched = []

    for item in ingredient_list:
        if not item or item.isspace():
            continue

        # Coba exact matching terlebih dahulu
        match = nutrition_df[nutrition_df['ingredient'].str.lower() == item.lower()]

        if not match.empty:
            row = match.iloc[0].to_dict()
            row['match_type'] = 'exact'
            row['match_score'] = 100
        else:
            # Jika tidak ada exact match, gunakan fuzzy matching
            match_row = find_best_match(item, nutrition_df)
            if match_row:
                row = match_row.copy()
                # Tambahkan informasi matching
                row['original_ingredient'] = item
                row['match_type'] = 'fuzzy'
            else:
                row = {
                    'ingredient': item,
                    'calories': 0,
                    'protein': 0,
                    'fat': 0,
                    'carbohydrates': 0,
                    'fiber': 0,
                    'calcium': 0,
                    'match_type': 'none'
                }
        enriched.append(row)

    return enriched


def process_dataset(recipe_df, nutrition_df):
    """Process the entire recipe dataset with nutrition information."""
    all_enriched = []

    for idx, row in recipe_df.iterrows():
        ingredient_text = row['Ingredients']
        ingredients = extract_ingredients(ingredient_text)

        # Filter out empty ingredients
        ingredients = [ing for ing in ingredients if ing and not ing.isspace()]

        enriched = enrich_ingredients(ingredients, nutrition_df)

        all_enriched.append({
            'ID': row['ID'],
            'Title': row['Title'],
            'Ingredients_Parsed': ingredients,
            'Ingredients_Enriched': enriched
        })

    return pd.DataFrame(all_enriched)