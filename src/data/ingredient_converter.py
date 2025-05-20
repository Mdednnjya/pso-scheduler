import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ingredient_converter')

# Dictionary of standard weights for common ingredients (in grams)
INGREDIENT_WEIGHTS = {
    # Protein sources
    'ayam': 100,  # per potong standar
    'daging': 100,  # per potong standar
    'ikan': 100,  # per potong standar
    'udang': 20,  # per ekor ukuran sedang
    'telur': 60,  # per butir
    'tahu': 80,  # per potong
    'tempe': 100,  # per potong standar

    # Vegetables
    'bawang putih': 5,  # per siung
    'bawang merah': 5,  # per siung
    'bawang bombay': 110,  # per buah ukuran sedang
    'tomat': 150,  # per buah ukuran sedang
    'cabai': 5,  # per buah
    'cabai rawit': 2,  # per buah
    'cabai merah': 10,  # per buah
    'wortel': 80,  # per buah ukuran sedang
    'kentang': 150,  # per buah ukuran sedang
    'labu siam': 300,  # per buah ukuran sedang
    'sawi': 100,  # per ikat kecil
    'kangkung': 100,  # per ikat kecil
    'bayam': 100,  # per ikat kecil
    'kacang panjang': 100,  # per ikat kecil

    # Herbs and spices
    'jahe': 15,  # per ruas
    'kunyit': 15,  # per ruas
    'lengkuas': 20,  # per ruas
    'serai': 15,  # per batang
    'daun salam': 1,  # per lembar
    'daun jeruk': 1,  # per lembar
    'kemiri': 5,  # per butir
    'ketumbar': 5,  # per sendok teh
    'merica': 5,  # per sendok teh
    'pala': 5,  # per sendok teh

    # Others
    'tepung': 10,  # per sendok makan
    'gula': 10,  # per sendok makan
    'garam': 5,  # per sendok teh
    'minyak': 15,  # per sendok makan
    'santan': 65,  # per sendok makan
    'kecap': 15,  # per sendok makan
    'nasi': 100,  # per mangkok kecil
    'beras': 70  # per mangkok kecil (raw)
}

# Dictionary for unit conversions to grams
UNIT_CONVERSIONS = {
    'kg': 1000,
    'g': 1,
    'gr': 1,
    'gram': 1,
    'ons': 100,
    'liter': 1000,  # Asumsi untuk bahan cair densitas = 1
    'l': 1000,
    'ml': 1,
    'cc': 1,
    'sdm': 15,  # sendok makan
    'sdt': 5,  # sendok teh
    'cup': 240,
    'gelas': 250,
    'mangkok': 300,
    'secukupnya': 10  # Default untuk "secukupnya"
}


def find_ingredient_base_weight(ingredient_name):
    """
    Find the base weight of an ingredient in grams

    Args:
        ingredient_name: Normalized ingredient name

    Returns:
        float: Base weight in grams, or None if not found
    """
    # Direct match first
    if ingredient_name in INGREDIENT_WEIGHTS:
        return INGREDIENT_WEIGHTS[ingredient_name]

    # Partial match for main ingredients
    for key, value in INGREDIENT_WEIGHTS.items():
        if key in ingredient_name:
            return value

    # Default weight if no match found
    return None


def convert_to_grams(ingredient):
    """
    Convert ingredient quantity and unit to grams

    Args:
        ingredient: Dict with quantity, unit, and name

    Returns:
        float: Weight in grams
    """
    quantity = ingredient['quantity']
    unit = ingredient['unit'].lower()
    name = ingredient['name']

    # No quantity or unit specified, use estimation
    if quantity == 0:
        quantity = 1.0

    # Direct unit conversion if possible
    if unit in UNIT_CONVERSIONS:
        weight_in_grams = quantity * UNIT_CONVERSIONS[unit]
        logger.debug(f"Converted {quantity} {unit} to {weight_in_grams}g using unit conversion")
        return weight_in_grams

    # Handle standard units like "buah", "siung", etc.
    standard_units = ['buah', 'bh', 'butir', 'siung', 'batang', 'btg', 'lembar', 'lbr', 'ikat']
    if unit in standard_units:
        base_weight = find_ingredient_base_weight(name)
        if base_weight:
            weight_in_grams = quantity * base_weight
            logger.debug(f"Converted {quantity} {unit} {name} to {weight_in_grams}g using standard weight")
            return weight_in_grams

    # Fall back to estimated weight based on ingredient name
    base_weight = find_ingredient_base_weight(name)
    if base_weight:
        # If no unit, assume the quantity is the number of standard portions
        weight_in_grams = quantity * base_weight
        logger.debug(f"Estimated {quantity} {name} to {weight_in_grams}g using ingredient base weight")
        return weight_in_grams

    # Last resort: default estimation
    default_weight = 50.0  # Default to 50g if we have no idea
    logger.warning(f"Using default weight estimation for {quantity} {unit} {name}: {default_weight}g")
    return default_weight


def get_ingredient_weight(ingredient):
    """
    Get the weight in grams for an ingredient

    Args:
        ingredient: Dict with quantity, unit, and name

    Returns:
        dict: Original ingredient with added weight_g field
    """
    try:
        weight_g = convert_to_grams(ingredient)

        # Add weight to ingredient dict
        result = ingredient.copy()
        result['weight_g'] = weight_g

        return result
    except Exception as e:
        logger.error(f"Error calculating weight for {ingredient}: {str(e)}")

        # Return original with estimated weight
        result = ingredient.copy()
        result['weight_g'] = 50.0  # Default fallback
        result['weight_estimated'] = True

        return result


def process_ingredient_weights(ingredients):
    """
    Process a list of ingredients to add weight in grams

    Args:
        ingredients: List of ingredient dictionaries

    Returns:
        list: Ingredients with weight_g added
    """
    processed = []

    for ingredient in ingredients:
        processed_ingredient = get_ingredient_weight(ingredient)
        processed.append(processed_ingredient)

    logger.info(f"Processed weights for {len(processed)} ingredients")
    return processed