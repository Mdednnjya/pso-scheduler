import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ingredient_converter')

# Dictionary of standard weights for common ingredients (in grams) - ENHANCED
INGREDIENT_WEIGHTS = {
    # Protein sources
    'ayam': 100,  # per potong standar
    'daging': 100,  # per potong standar
    'daging sapi': 100,  # per potong standar
    'ikan': 100,  # per potong standar
    'udang': 20,  # per ekor ukuran sedang
    'telur': 60,  # per butir
    'telor': 60,  # per butir
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

    # Herbs and spices - ENHANCED
    'jahe': 15,  # per ruas
    'kunyit': 15,  # per ruas
    'lengkuas': 20,  # per ruas
    'serai': 15,  # per batang
    'sereh': 15,  # per batang
    'daun salam': 1,  # per lembar
    'daun jeruk': 1,  # per lembar
    'kemiri': 5,  # per butir
    'ketumbar': 5,  # per sendok teh
    'merica': 1,  # per sendok teh or per butir - FIXED
    'lada': 1,  # per sendok teh or per butir - FIXED
    'pala': 5,  # per sendok teh

    # Others
    'tepung': 100,  # per 100g default
    'tepung terigu': 100,  # per 100g
    'tepung tapioka': 100,  # per 100g - ADDED
    'gula': 10,  # per sendok makan
    'garam': 5,  # per sendok teh
    'minyak': 15,  # per sendok makan
    'santan': 65,  # per sendok makan
    'kecap': 15,  # per sendok makan
    'nasi': 100,  # per mangkok kecil
    'beras': 70,  # per mangkok kecil (raw)

    # Special cases for problematic ingredients
    'kepala kambing': 2000,  # ADDED - realistic weight
    'kepala': 1000,  # generic head weight
}

# Enhanced kamus khusus untuk bumbu-bumbu ringan dan instruksi
LIGHT_INGREDIENTS = {
    'daun salam': 1,  # 1g per lembar
    'daun jeruk': 1,  # 1g per lembar
    'daun pandan': 1,  # 1g per lembar
    'daun kunyit': 1,  # 1g per lembar
    'daun kemangi': 5,  # 5g per ikat kecil
    'daun bawang': 10,  # 10g per batang
    'daun seledri': 5,  # 5g per batang
    'daun mint': 2,  # 2g per tangkai
    'daun kari': 1,  # 1g per lembar
    'serai': 15,  # 15g per batang
    'sereh': 15,  # 15g per batang
    'cabe rawit': 2,  # 2g per buah
    'cabai rawit': 2,  # 2g per buah
    'cengkeh': 1,  # 1g per buah
    'kemiri': 5,  # 5g per butir
    'pala': 5,  # 5g per butir
    'merica': 1,  # 1g per sdt - FIXED
    'lada': 1,  # 1g per sdt - FIXED
    'ketumbar': 5,  # 5g per sdt
    'kayu manis': 5,  # 5g per batang
    'kapulaga': 1,  # 1g per buah
    'bunga lawang': 1,  # 1g per buah
    'bunga pekak': 1,  # 1g per buah
    'kuncit': 15,  # 15g per ruas
    'kunyit': 15,  # 15g per ruas
    'jahe': 15,  # 15g per ruas
    'lengkuas': 20,  # 20g per ruas
    'kencur': 10,  # 10g per ruas
    'terasi': 5,  # 5g per sdt
    'petis': 10,  # 10g per sdm

    # Instructions (0 weight)
    'bakar': 0, 'iris': 0, 'potong': 0, 'cincang': 0, 'haluskan': 0,
    'aduk': 0, 'campur': 0, 'geprek': 0, 'tumbuk': 0, 'rebus': 0,
    'kukus': 0, 'goreng': 0, 'tumis': 0, 'belah': 0, 'buang': 0,
    'ambil': 0, 'cuci': 0, 'bahan': 0, 'pelengkap': 0, 'tambahan': 0,
    'utama': 0, 'bumbu': 0, 'dihaluskan': 0, 'digeprek': 0, 'diiris': 0
}

# ENHANCED Dictionary for unit conversions to grams
UNIT_CONVERSIONS = {
    'kg': 1000,
    'g': 1,
    'gr': 1,  # FIXED: gr = gram, not confusion
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
    'lembar': 1,  # 1g per lembar (for leaves)
    'lbr': 1,  # Singkatan lembar
    'batang': 15,  # Untuk serai, dll
    'btg': 15,  # Singkatan batang
    'buah': 50,  # Ukuran sedang (bisa bervariasi)
    'bh': 50,  # Singkatan buah
    'butir': 60,  # Default untuk telur, WILL BE OVERRIDDEN by ingredient-specific
    'btr': 60,  # Singkatan butir
    'siung': 5,  # Untuk bawang
    'ruas': 15,  # Untuk jahe, kunyit, dll
    'seruas': 15,  # Untuk jahe, kunyit, dll
    'cm': 5,  # Estimasi per cm untuk rempah
    'senti': 5,  # Estimasi per cm untuk rempah
    'biji': 5,  # Default, WILL BE OVERRIDDEN by ingredient-specific
    'ikat': 100,  # Untuk sayuran ikat
    'ikt': 100,  # Singkatan ikat
    'potong': 100,  # Untuk daging/ayam
    'ptg': 100,  # Singkatan potong
    'iris': 10,  # Untuk irisan
    'ekor': 100,  # Default animal weight, WILL BE OVERRIDDEN
    'pack': 50,  # Pack/bungkus
    'bungkus': 50,
    'sachet': 10,
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
    # Periksa apakah ini bumbu ringan atau instruksi
    for key, value in LIGHT_INGREDIENTS.items():
        if key in ingredient_name.lower():
            return value

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
    ENHANCED conversion with intelligent validation and correction

    Args:
        ingredient: Dict with quantity, unit, and name

    Returns:
        float: Weight in grams
    """
    quantity = ingredient['quantity']
    unit = ingredient['unit'].lower() if isinstance(ingredient['unit'], str) else ""
    name = ingredient['name']

    # Check if it's an instruction (not an ingredient)
    for key in LIGHT_INGREDIENTS:
        if key == name.lower() and LIGHT_INGREDIENTS[key] == 0:
            logger.debug(f"Detected instruction, not ingredient: {name}")
            return 0

    # No quantity or unit specified, use estimation
    if quantity == 0:
        quantity = 1.0

    # ENHANCED: Handle special ingredient-specific units first
    weight_g = handle_ingredient_specific_units(ingredient)
    if weight_g is not None:
        return validate_and_correct_weight(name, unit, quantity, weight_g)

    # Direct unit conversion if possible
    if unit in UNIT_CONVERSIONS:
        # Special case for 'cm' or 'senti'
        if unit == 'cm' or unit == 'senti':
            if 'jahe' in name.lower() or 'kunyit' in name.lower() or 'lengkuas' in name.lower() or 'kencur' in name.lower():
                weight_in_grams = quantity * UNIT_CONVERSIONS[unit]
                logger.debug(f"Converted {quantity} {unit} of {name} to {weight_in_grams}g")
                return validate_and_correct_weight(name, unit, quantity, weight_in_grams)

        # Special case for 'lembar' to avoid confusion with 'liter'
        if unit == 'lembar' or unit == 'lbr':
            weight_in_grams = quantity * UNIT_CONVERSIONS[unit]
            if 'daun' in name.lower():  # Jika ini daun, pastikan berat per lembar 1g
                logger.debug(f"Converting leaf: {quantity} {unit} of {name} to {weight_in_grams}g")
                return validate_and_correct_weight(name, unit, quantity, weight_in_grams)

        # Regular unit conversion
        weight_in_grams = quantity * UNIT_CONVERSIONS[unit]
        logger.debug(f"Converted {quantity} {unit} to {weight_in_grams}g using unit conversion")
        return validate_and_correct_weight(name, unit, quantity, weight_in_grams)

    # Handle standard units like "buah", "siung", etc.
    standard_units = ['buah', 'bh', 'butir', 'siung', 'batang', 'btg', 'lembar', 'lbr', 'ikat']
    if unit in standard_units:
        base_weight = find_ingredient_base_weight(name)
        if base_weight:
            weight_in_grams = quantity * base_weight
            logger.debug(f"Converted {quantity} {unit} {name} to {weight_in_grams}g using standard weight")
            return validate_and_correct_weight(name, unit, quantity, weight_in_grams)

    # Fall back to estimated weight based on ingredient name
    base_weight = find_ingredient_base_weight(name)
    if base_weight:
        # If no unit, assume the quantity is the number of standard portions
        weight_in_grams = quantity * base_weight
        logger.debug(f"Estimated {quantity} {name} to {weight_in_grams}g using ingredient base weight")
        return validate_and_correct_weight(name, unit, quantity, weight_in_grams)

    # Last resort: intelligent default estimation
    return estimate_default_weight(name, quantity)


def handle_ingredient_specific_units(ingredient):
    """
    ADDED: Handle ingredient-specific unit conversions

    Args:
        ingredient: Dict with quantity, unit, and name

    Returns:
        float or None: Weight in grams if handled, None otherwise
    """
    quantity = ingredient['quantity']
    unit = ingredient['unit'].lower()
    name = ingredient['name'].lower()

    # Ingredient-specific unit handling
    specific_conversions = {
        # Telur conversions - CRITICAL FIX
        'telur': {
            'butir': 60, 'biji': 60, 'buah': 60, '': 60  # Empty unit defaults to butir
        },
        'telor': {
            'butir': 60, 'biji': 60, 'buah': 60, '': 60
        },

        # Merica conversions - CRITICAL FIX
        'merica': {
            'butir': 1, 'biji': 1, 'sdt': 5, '': 5  # Default to sdt
        },
        'lada': {
            'butir': 1, 'biji': 1, 'sdt': 5, '': 5
        },

        # Udang conversions - CRITICAL FIX
        'udang': {
            'ekor': 20, 'g': 1, 'gr': 1, 'gram': 1, '': 100  # Default 100g if no unit
        },

        # Bawang conversions
        'bawang putih': {
            'siung': 5, 'buah': 50, 'g': 1, 'gr': 1
        },
        'bawang merah': {
            'siung': 5, 'buah': 30, 'g': 1, 'gr': 1
        },

        # Kepala kambing - CRITICAL FIX
        'kepala kambing': {
            'buah': 2000, '': 2000  # 2kg for goat head
        },
        'kepala': {
            'buah': 1000, '': 1000  # 1kg generic
        }
    }

    # Check for specific ingredient matches
    for ingredient_key, unit_map in specific_conversions.items():
        if ingredient_key in name:
            if unit in unit_map:
                weight = quantity * unit_map[unit]
                logger.debug(f"Ingredient-specific conversion: {quantity} {unit} {name} = {weight}g")
                return weight
            # If unit not found, try default (empty string key)
            elif '' in unit_map:
                weight = quantity * unit_map['']
                logger.debug(f"Using default weight for {name}: {quantity} units = {weight}g")
                return weight

    return None  # Not handled by ingredient-specific logic


def validate_and_correct_weight(name, unit, quantity, calculated_weight):
    """
    ADDED: Validate calculated weight and apply corrections if unrealistic
    """
    name_lower = name.lower()
    corrections = []

    # Realistic weight bounds for different ingredient categories
    weight_bounds = {
        'spices': {'min': 0.1, 'max': 50, 'keywords': ['merica', 'lada', 'ketumbar', 'jintan', 'pala', 'cengkeh']},
        'herbs': {'min': 0.5, 'max': 30, 'keywords': ['daun', 'sereh', 'serai', 'jahe', 'kunyit', 'lengkuas']},
        'proteins': {'min': 50, 'max': 3000,
                     'keywords': ['daging', 'ayam', 'ikan', 'udang', 'telur', 'sapi', 'kambing', 'kepala']},
        'vegetables': {'min': 10, 'max': 2000, 'keywords': ['bawang', 'tomat', 'kentang', 'labu', 'wortel', 'kol']},
        'starches': {'min': 10, 'max': 1000, 'keywords': ['tepung', 'tapioka', 'terigu', 'maizena', 'beras']}
    }

    # Check bounds for each category
    for category, bounds in weight_bounds.items():
        if any(keyword in name_lower for keyword in bounds['keywords']):
            if calculated_weight < bounds['min'] or calculated_weight > bounds['max']:
                # Apply intelligent correction
                if calculated_weight > bounds['max']:
                    # Too heavy - likely unit conversion error
                    if calculated_weight > bounds['max'] * 10:
                        corrected_weight = calculated_weight / 1000  # kg -> g conversion
                        if bounds['min'] <= corrected_weight <= bounds['max']:
                            corrections.append(
                                f"Weight {calculated_weight}g too high, corrected to {corrected_weight}g")
                            calculated_weight = corrected_weight
                elif calculated_weight < bounds['min']:
                    # Too light - likely missing quantity or unit error
                    if category == 'proteins' and calculated_weight < 10:
                        corrected_weight = calculated_weight * 100  # 2g -> 200g
                        if bounds['min'] <= corrected_weight <= bounds['max']:
                            corrections.append(f"Weight {calculated_weight}g too low, corrected to {corrected_weight}g")
                            calculated_weight = corrected_weight
            break

    if calculated_weight > 5000:  # > 5kg per ingredient
        logger.warning(f"Extreme weight detected: {calculated_weight}g for '{name}'")

        # Special handling for flour/tepung
        if any(flour_word in name_lower for flour_word in ['tepung', 'flour', 'terigu']):
            # Tepung: max reasonable = 1kg per recipe
            if calculated_weight > 1000:
                corrected_weight = min(calculated_weight, 1000)
                corrections.append(f"Capped flour weight from {calculated_weight}g to {corrected_weight}g")
                calculated_weight = corrected_weight

        # Special handling for meat/daging
        elif any(meat_word in name_lower for meat_word in ['daging', 'sapi', 'ayam', 'kambing']):
            # Daging: max reasonable = 2kg per recipe
            if calculated_weight > 2000:
                corrected_weight = min(calculated_weight, 2000)
                corrections.append(f"Capped meat weight from {calculated_weight}g to {corrected_weight}g")
                calculated_weight = corrected_weight

        # Generic extreme case
        else:
            corrected_weight = calculated_weight / 10  # Likely unit error
            if corrected_weight > 50:  # Still reasonable
                corrections.append(f"Extreme value correction: {calculated_weight}g → {corrected_weight}g")
                calculated_weight = corrected_weight

    if corrections:
        logger.info(f"Weight corrections for '{name}': {corrections}")

    return calculated_weight


def estimate_default_weight(name, quantity):
    """
    ADDED: Intelligent default weight estimation based on ingredient category
    """
    name_lower = name.lower()

    # Category-based defaults
    if any(spice in name_lower for spice in ['bumbu', 'rempah', 'bubuk', 'merica', 'lada']):
        return max(quantity * 5, 2)  # Minimum 2g for spices
    elif any(herb in name_lower for herb in ['daun', 'sereh', 'jahe', 'kunyit']):
        return max(quantity * 10, 5)  # Minimum 5g for herbs
    elif any(protein in name_lower for protein in ['daging', 'ayam', 'ikan', 'udang', 'telur', 'kepala']):
        return max(quantity * 100, 50)  # Minimum 50g for proteins
    elif any(veg in name_lower for veg in ['bawang', 'tomat', 'kentang', 'sayur']):
        return max(quantity * 50, 20)  # Minimum 20g for vegetables
    else:
        return max(quantity * 30, 10)  # Conservative default


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

        # ENHANCED Verifikasi berat yang tidak masuk akal
        if weight_g > 1000:
            # Check if this is a legitimate heavy ingredient
            heavy_ingredients = ['daging', 'ayam', 'ikan', 'beras', 'tepung', 'kepala', 'kambing', 'sapi']
            is_heavy_ingredient = any(heavy in ingredient['name'].lower() for heavy in heavy_ingredients)

            if not is_heavy_ingredient:
                logger.warning(
                    f"Unrealistic weight detected: {weight_g}g for {ingredient['raw']}. Attempting correction.")

                # Koreksi khusus untuk daun dengan unit 'l' yang seharusnya 'lembar'
                if ingredient['unit'].lower() == 'l' and 'daun' in ingredient['name'].lower():
                    corrected_weight = ingredient['quantity'] * 1  # 1g per lembar
                    logger.info(f"Corrected weight from {weight_g}g to {corrected_weight}g (l -> lembar)")
                    weight_g = corrected_weight
                # Koreksi umum untuk berat berlebihan
                else:
                    # Coba kurangi dengan faktor 1000 (kg -> g, l -> ml)
                    corrected_weight = weight_g / 1000
                    logger.info(f"Corrected weight from {weight_g}g to {corrected_weight}g (generic correction)")
                    weight_g = corrected_weight

        # Add weight to ingredient dict
        result = ingredient.copy()
        result['weight_g'] = weight_g

        return result
    except Exception as e:
        logger.error(f"Error calculating weight for {ingredient}: {str(e)}")

        # Return original with estimated weight
        result = ingredient.copy()
        result['weight_g'] = 30.0  # Default fallback yang lebih konservatif
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