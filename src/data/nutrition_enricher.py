import pandas as pd
import numpy as np
from fuzzywuzzy import process, fuzz
import logging
from src.data.ingredient_parser import extract_ingredients, normalize_ingredient_name
from src.data.ingredient_converter import process_ingredient_weights

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('nutrition_enricher')

# Sinonim untuk membantu matching
INGREDIENT_SYNONYMS = {
    'ayam': ['chicken', 'daging ayam', 'ayam potong', 'ayam kampung', 'ayam broiler', 'ayam negeri', 'ayam fillet'],
    'bawang putih': ['garlic', 'bawang puteh', 'bwang putih', 'bawang putih kupas', 'baput', 'bawput'],
    'bawang merah': ['shallot', 'bawang bawang', 'bamer', 'bawang merah kupas', 'bw merah', 'bawmer'],
    'bawang bombay': ['onion', 'bombay', 'bawang bombai', 'bawang bombe', 'bombai', 'bawang onion'],
    'cabai': ['cabe', 'chili', 'chilli', 'cabai', 'lombok', 'cabe', 'cabai merah', 'cabe merah', 'cabe rawit'],
    'cabai rawit': ['cabe rawit', 'chili', 'rawit', 'cabe kecil', 'cabai kecil', 'lombok rawit', 'rica', 'rica-rica'],
    'cabai merah': ['cabe merah', 'red chili', 'cabe besar', 'cabai besar', 'lombok merah', 'cabai keriting'],
    'daging sapi': ['beef', 'daging', 'sapi', 'daging lembu', 'has dalam', 'has luar', 'tenderloin', 'sirloin', 'tetelan'],
    'daging kambing': ['lamb', 'kambing', 'domba', 'daging domba', 'mutton', 'daging kambink'],
    'udang': ['shrimp', 'prawn', 'udang windu', 'udang galah', 'udang vaname', 'udang pancet', 'udang besar', 'udang kecil'],
    'ikan': ['fish', 'ikan laut', 'ikan air tawar', 'seafood', 'ikan kakap', 'ikan nila', 'ikan gurame', 'ikan tongkol'],
    'telur': ['egg', 'telor', 'telur ayam', 'telur bebek', 'telur puyuh', 'telur itik', 'telor ayam', 'telur kampung', 'telur asin', 'telor asin'],
    'tahu': ['tofu', 'bean curd', 'tahu putih', 'tahu kuning', 'tahu cina', 'tahu sutra', 'tahu susu', 'tahu jepang'],
    'tempe': ['tempeh', 'fermented soybean', 'tempe kedelai', 'tempe gembus', 'tempe mendoan', 'tempe goreng'],
    'jahe': ['ginger', 'jahey', 'halia', 'jahe merah', 'jahe emprit', 'jahe gajah'],
    'kunyit': ['turmeric', 'kunir', 'kunyet', 'kunyit tuha', 'kunyit putih', 'kunyit bubuk'],
    'kencur': ['galangal', 'lesser galangal', 'kaempferia galanga', 'kencor', 'cekur'],
    'lengkuas': ['galangal', 'greater galangal', 'laos', 'langkuas', 'lengkueh'],
    'minyak': ['oil', 'minyak goreng', 'minyak sayur', 'minyak zaitun', 'olive oil', 'minyak kelapa', 'minyak jagung', 'minyak canola'],
    'gula': ['sugar', 'gula pasir', 'gula jawa', 'gula aren', 'gula merah', 'gula palem', 'gula batu', 'gula kelapa'],
    'garam': ['salt', 'garam dapur', 'garam laut', 'himalayan salt', 'garam himalaya', 'garam meja'],
    'tepung': ['flour', 'tepung terigu', 'tepung beras', 'tepung kanji', 'tepung maizena', 'tepung tapioka', 'tepung cakra', 'tepung segitiga', 'tepung gandum'],
    'santan': ['coconut milk', 'santan kelapa', 'coconut cream', 'santan kental', 'santan encer', 'santan instan'],
    'tomat': ['tomato', 'tomat merah', 'tomat hijau', 'tomat cherry', 'tomat sayur', 'tomato', 'buah tomat'],
    'wortel': ['carrot', 'baby carrot', 'wortel import', 'wortel lokal', 'karrot'],
    'kentang': ['potato', 'ubi kentang', 'kentang dieng', 'kentang granola', 'kentang putih', 'kentang merah'],
    'daun salam': ['bay leaf', 'indonesian bay leaf', 'salam leaf', 'daun salam kering', 'salam'],
    'daun jeruk': ['kaffir lime leaf', 'lime leaf', 'jeruk purut', 'daun jeruk purut', 'daun limau'],
    'serai': ['lemongrass', 'sereh', 'sarai', 'sereh dapur', 'batang serai', 'batang sereh'],
    'kacang': ['peanut', 'bean', 'nut', 'kacang tanah', 'kacang hijau', 'kacang merah', 'kacang polong', 'kacang almond', 'kacang mete'],
    'terasi': ['shrimp paste', 'belacan', 'trassi', 'terasi udang', 'terasi ikan', 'petis'],
    'kemiri': ['candlenut', 'candleberry', 'kukui nut', 'kemiri sangrai'],
    'kemangi': ['lemon basil', 'thai basil', 'daun kemangi', 'basil', 'sweet basil'],
    'pala': ['nutmeg', 'mace', 'buah pala', 'biji pala', 'pala bubuk'],
    'cengkeh': ['clove', 'bunga cengkeh', 'cengkih', 'cengkeh kering'],
    'ketumbar': ['coriander', 'coriander seed', 'cilantro seed', 'ketumbar bubuk', 'ketumbar utuh'],
    'merica': ['pepper', 'black pepper', 'lada', 'lada hitam', 'lada putih', 'merica bubuk', 'lada bubuk'],
    'pete': ['petai', 'stink bean', 'sataw', 'papan pete', 'pete kupas', 'petai segar'],
    'baby corn': ['baby jagung', 'putren', 'jagung muda kecil', 'jagung putren', 'jagung baby', 'jagung mini'],
    'sawi': ['mustard greens', 'sawi hijau', 'sawi putih', 'pakcoy', 'caisim', 'pokchoy', 'bok choy'],
    'kaldu bubuk': ['penyedap', 'royko', 'masako', 'maggi', 'kaldu jamur', 'bumbu penyedap', 'kaldu sapi', 'kaldu ayam'],
    'air': ['water', 'air putih', 'air bersih', 'air matang', 'air mineral', 'air hangat', 'air panas'],
    'saus': ['sauce', 'saos', 'saus tomat', 'saus cabai', 'saus tiram', 'saus asam manis', 'saus sambal'],
    'kecap': ['soy sauce', 'kecap manis', 'kecap asin', 'sweet soy sauce', 'black soy sauce', 'kecap ikan', 'kecap jamur'],
    'bumbu': ['seasoning', 'bumbu dapur', 'bumbu rempah', 'bumbu halus', 'bumbu racik', 'bumbu instan', 'bumbu gule', 'bumbu rendang'],
    'jeruk nipis': ['lime', 'limau', 'citrus', 'jeruk limau', 'jeruk nipis segar', 'lime juice'],
    'kelapa': ['coconut', 'kelapa parut', 'kelapa muda', 'kelapa tua', 'coconut flesh', 'coconut grated'],
    'daun bawang': ['scallion', 'green onion', 'spring onion', 'leek', 'daun bawang pre', 'bawang prey', 'prey', 'bawang daun'],
    'terigu': ['wheat flour', 'tepung terigu', 'tepung gandum', 'tepung cakra', 'tepung segitiga', 'tepung serbaguna'],
    'adonan': ['dough', 'batter', 'adonan kue', 'adonan basah', 'adonan kering', 'adonan isian', 'adonan tepung']
}


def expand_synonyms(ingredient_name):
    """
    Expand ingredient name with synonyms for better matching
    """
    expanded = [ingredient_name]

    # Add synonyms if available
    for key, synonyms in INGREDIENT_SYNONYMS.items():
        if key in ingredient_name.lower():
            expanded.extend(synonyms)
            break

    # Also check if ingredient is a synonym and add the main term
    for key, synonyms in INGREDIENT_SYNONYMS.items():
        if any(synonym.lower() in ingredient_name.lower() for synonym in synonyms):
            expanded.append(key)
            break

    return expanded


def find_best_match(ingredient, nutrition_df, threshold=70):  # Naikkan threshold dari 65 menjadi 70
    """
    Find the best match for an ingredient in the nutrition database

    Args:
        ingredient: Dict with ingredient info
        nutrition_df: DataFrame with nutrition data
        threshold: Minimum score to consider a match (default: 70)

    Returns:
        dict: Match result with nutrition data and score
    """
    if not ingredient or not ingredient.get('name'):
        return None

    # Normalize the ingredient name
    normalized_ingredient = ingredient['name']

    # Skip instruksi bukan bahan
    instruction_words = [
        'iris', 'potong', 'cincang', 'bakar', 'rebus', 'kukus', 'goreng',
        'tumis', 'belah', 'aduk', 'campur', 'ambil', 'buang', 'haluskan',
        'bahan', 'cincang', 'tumbuk', 'geprek', 'sangrai', 'seduh', 'rendam'
    ]

    # Cek apakah nama bahan hanya berisi instruksi
    if any(normalized_ingredient.lower() == word for word in instruction_words):
        logger.warning(f"Ingredient appears to be instruction: {normalized_ingredient}")
        return {
            'ingredient': normalized_ingredient,
            'original_ingredient': ingredient['raw'],
            'match_score': 0,
            'match_type': 'instruction',
            'calories': 0,
            'protein': 0,
            'fat': 0,
            'carbohydrates': 0,
            'fiber': 0,
            'calcium': 0
        }

    blacklist_matches = {
        'merica': ['coklat', 'cokla', 'chocolate'],
        'gula': ['gulai', 'gulei'],
        'bakar': ['bakwan', 'bakmi', 'bakso'],
        'jahe': ['geplak', 'kue'],
        'bawang': ['tawang', 'kawang'],
        'garam': ['daun', 'talas', 'salam'],
        'ikan': ['bukan', 'makan', 'minum'],
        'dadu': ['madu'],
        'air': ['dair', 'hair', 'fair'],
        'lada': ['soda', 'jada']
    }

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

    # Filter matches berdasarkan blacklist
    for bad_word, avoid_words in blacklist_matches.items():
        if bad_word in normalized_ingredient.lower():
            # Hapus hasil match yang mengandung kata yang harus dihindari
            filtered_matches = []
            for match in best_matches:
                match_name = match[0].lower()
                if not any(avoid in match_name for avoid in avoid_words):
                    filtered_matches.append(match)

            # Gunakan filtered matches jika ada, jika tidak gunakan best_matches asli
            if filtered_matches:
                best_matches = filtered_matches
                break

    # Find the best among all variations
    if best_matches:
        best_match = max(best_matches, key=lambda x: x[1])
        match_name, score = best_match

        # log untuk debugging matching
        logger.debug(f"Match for '{normalized_ingredient}': '{match_name}' with score {score}")

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

    # Flag untuk mencatat jika ada masalah dengan perhitungan nutrisi
    has_problematic_ingredients = False
    problematic_ingredients = []

    for ing in enriched_ingredients:
        # Skip instruksi yang bukan bahan
        if ing.get('match_type') == 'instruction':
            continue

        # Cek apakah bahan ini memiliki nilai nutrisi yang tidak realistis
        if ing.get('calories', 0) > 1000:  # Batas kalori per bahan yang masuk akal
            has_problematic_ingredients = True
            problematic_ingredients.append({
                'name': ing.get('original_ingredient', ''),
                'calories': ing.get('calories', 0),
                'weight': ing.get('parsed_ingredient', {}).get('weight_g', 0)
            })

        # Sum nutrition values
        for nutrient in total_nutrition:
            try:
                value = float(ing.get(nutrient, 0) or 0)
                total_nutrition[nutrient] += value
            except (ValueError, TypeError):
                pass

    # Koreksi untuk bahan bermasalah (nilai kalori terlalu tinggi)
    if has_problematic_ingredients:
        logger.warning(f"Found problematic ingredients with unrealistic nutrition values: {problematic_ingredients}")

        # Khusus untuk daun salam dan daun jeruk yang sering salah unit
        for ing in enriched_ingredients:
            ing_name = ing.get('parsed_ingredient', {}).get('name', '').lower()
            ing_unit = ing.get('parsed_ingredient', {}).get('unit', '').lower()

            # Deteksi kesalahan unit lembar menjadi liter
            if ('daun salam' in ing_name or 'daun jeruk' in ing_name) and ing_unit == 'l':
                original_weight = ing.get('parsed_ingredient', {}).get('weight_g', 0)

                if original_weight > 10:  # Terlalu berat untuk daun
                    # Koreksi: asumsi yang dimaksud adalah lembar, bukan liter
                    corrected_weight = ing.get('parsed_ingredient', {}).get('quantity', 1) * 1.0  # 1g per lembar
                    scaling_factor = corrected_weight / original_weight if original_weight > 0 else 0

                    logger.info(
                        f"Correcting weight for {ing.get('original_ingredient')}: {original_weight}g -> {corrected_weight}g")

                    # Koreksi nilai nutrisi
                    for nutrient in total_nutrition:
                        if nutrient in ing:
                            # Kurangi dari total
                            total_nutrition[nutrient] -= ing[nutrient]
                            # Skala ulang nilai nutrisi
                            ing[nutrient] *= scaling_factor
                            # Tambahkan yang sudah dikoreksi ke total
                            total_nutrition[nutrient] += ing[nutrient]

                    # Update berat dalam data ingredient
                    ing['parsed_ingredient']['weight_g'] = corrected_weight
                    ing['parsed_ingredient']['unit'] = 'lembar'  # Perbaiki unit

    # Validasi total nutrisi
    # Jika total kalori masih tidak masuk akal setelah koreksi
    if total_nutrition['calories'] > 10000:  # Batas atas yang sangat konservatif
        logger.warning(
            f"Total calories still unrealistic after corrections: {total_nutrition['calories']}. Applying final correction.")

        # Hitung faktor skala untuk normalisasi
        scaling_factor = 3000 / total_nutrition['calories'] if total_nutrition[
                                                                   'calories'] > 0 else 0  # Target ~3000 kalori

        # Terapkan ke semua nilai nutrisi
        for nutrient in total_nutrition:
            total_nutrition[nutrient] *= scaling_factor

        logger.info(f"Applied global scaling factor of {scaling_factor} to normalize nutrition values")

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


def process_dataset(recipe_df, nutrition_df, supplement_df=None):
    """
    Process the entire recipe dataset

    Args:
        recipe_df: DataFrame with recipe data
        nutrition_df: DataFrame with nutrition data
        supplement_df: Optional DataFrame with supplementary nutrition data

    Returns:
        DataFrame: Processed recipes
    """
    # Gabungkan database nutrisi jika ada supplement
    if supplement_df is not None:
        # Pastikan tidak ada duplikat
        combined_nutrition = pd.concat([nutrition_df, supplement_df]).drop_duplicates(subset=['ingredient'])
        logger.info(f"Combined nutrition database has {len(combined_nutrition)} ingredients (added {len(supplement_df)} supplements)")
        nutrition_df = combined_nutrition
    else:
        logger.info(f"Using original nutrition database with {len(nutrition_df)} ingredients")

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