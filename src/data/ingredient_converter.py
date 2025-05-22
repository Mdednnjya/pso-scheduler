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

# Tambahkan kamus khusus untuk bumbu-bumbu ringan dan instruksi
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
    'cabe rawit': 2,  # 2g per buah
    'cabai rawit': 2,  # 2g per buah
    'cengkeh': 1,  # 1g per buah
    'kemiri': 5,  # 5g per butir
    'pala': 5,  # 5g per butir
    'merica': 1,  # 1g per sdt
    'lada': 1,  # 1g per sdt
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
    # Tambahkan instruksi yang bukan bahan dengan berat 0
    'bakar': 0,  # Instruksi, bukan bahan
    'iris': 0,  # Instruksi, bukan bahan
    'potong': 0,  # Instruksi, bukan bahan
    'cincang': 0,  # Instruksi, bukan bahan
    'haluskan': 0,  # Instruksi, bukan bahan
    'aduk': 0,  # Instruksi, bukan bahan
    'campur': 0,  # Instruksi, bukan bahan
    'geprek': 0,  # Instruksi, bukan bahan
    'tumbuk': 0,  # Instruksi, bukan bahan
    'rebus': 0,  # Instruksi, bukan bahan
    'kukus': 0,  # Instruksi, bukan bahan
    'goreng': 0,  # Instruksi, bukan bahan
    'tumis': 0,  # Instruksi, bukan bahan
    'belah': 0,  # Instruksi, bukan bahan
    'buang': 0,  # Instruksi, bukan bahan
    'ambil': 0,  # Instruksi, bukan bahan
    'cuci': 0,  # Instruksi, bukan bahan
    'bahan': 0,  # Label kategori, bukan bahan
    'pelengkap': 0,  # Label kategori, bukan bahan
    'tambahan': 0,  # Label kategori, bukan bahan
    'utama': 0,  # Label kategori, bukan bahan
    'bumbu': 0,  # Label kategori, bukan bahan
    'dihaluskan': 0,  # Instruksi, bukan bahan
    'digeprek': 0,  # Instruksi, bukan bahan
    'diiris': 0  # Instruksi, bukan bahan
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
    'lembar': 1,  # Penting: 1g per lembar, bukan liter!
    'lbr': 1,  # Singkatan lembar
    'batang': 15,  # Untuk serai, dll
    'btg': 15,  # Singkatan batang
    'buah': 50,  # Ukuran sedang (bisa bervariasi)
    'bh': 50,  # Singkatan buah
    'butir': 60,  # Untuk telur, dll
    'btr': 60,  # Singkatan butir
    'siung': 5,  # Untuk bawang
    'ruas': 15,  # Untuk jahe, kunyit, dll
    'seruas': 15,    # Untuk jahe, kunyit, dll
    'cm': 5,         # Estimasi per cm untuk rempah
    'senti': 5,      # Estimasi per cm untuk rempah
    'biji': 5,       # Untuk rempah-rempah tertentu
    'ikat': 100,  # Untuk sayuran ikat
    'ikt': 100,  # Singkatan ikat
    'potong': 100,  # Untuk daging/ayam
    'ptg': 100,  # Singkatan potong
    'iris': 10,  # Untuk irisan
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
    Convert ingredient quantity and unit to grams

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

    # Direct unit conversion if possible
    if unit in UNIT_CONVERSIONS:
        # Special case for 'cm' or 'senti'
        if unit == 'cm' or unit == 'senti':
            if 'jahe' in name.lower() or 'kunyit' in name.lower() or 'lengkuas' in name.lower() or 'kencur' in name.lower():
                weight_in_grams = quantity * UNIT_CONVERSIONS[unit]
                logger.debug(f"Converted {quantity} {unit} of {name} to {weight_in_grams}g")
                return weight_in_grams

        # Special case for 'lembar' to avoid confusion with 'liter'
        if unit == 'lembar' or unit == 'lbr':
            weight_in_grams = quantity * UNIT_CONVERSIONS[unit]
            if 'daun' in name.lower():  # Jika ini daun, pastikan berat per lembar 1g
                logger.debug(f"Converting leaf: {quantity} {unit} of {name} to {weight_in_grams}g")
                return weight_in_grams

        # Special case for 'l' to avoid misinterpreting as 'liter'
        if unit == 'l' and ('daun' in name.lower() or 'lembar' in name.lower()):
            # Kemungkinan ini adalah 'lembar', bukan 'liter'
            logger.warning(f"Detected potential unit confusion: '{unit}' in '{name}'. Treating as 'lembar' not 'liter'")
            return quantity * 1  # 1g per lembar

        # Special case for 'biji' or 'butir'
        if unit == 'biji' or unit == 'butir':
            if 'kemiri' in name.lower():
                return quantity * 5  # 5g per butir kemiri
            elif 'telur' in name.lower():
                return quantity * 60  # 60g per telur
            elif 'cengkeh' in name.lower() or 'kapulaga' in name.lower() or 'bunga' in name.lower():
                return quantity * 1  # 1g per butir rempah

        # Regular unit conversion
        weight_in_grams = quantity * UNIT_CONVERSIONS[unit]
        logger.debug(f"Converted {quantity} {unit} to {weight_in_grams}g using unit conversion")
        return weight_in_grams

    # Handle special vocabulary like 'ruas' or 'seruas'
    if 'ruas' in name.lower() or 'seruas' in name.lower():
        # Estimasi berat ruas untuk rempah
        if 'jahe' in name.lower():
            return 15.0 * quantity
        elif 'kunyit' in name.lower():
            return 15.0 * quantity
        elif 'lengkuas' in name.lower() or 'laos' in name.lower():
            return 20.0 * quantity
        elif 'kencur' in name.lower():
            return 10.0 * quantity

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

    # Last resort: default estimation based on ingredient category
    if any(spice in name.lower() for spice in ['bumbu', 'rempah', 'bubuk', 'merica', 'ketumbar', 'pala']):
        default_weight = 5.0  # Default for spices
    elif 'daun' in name.lower():
        default_weight = 1.0  # Default for leaves
    elif any(category in name.lower() for category in ['bahan', 'pelengkap', 'utama', 'dihaluskan']):
        default_weight = 0.0  # Ini hanya label kategori
    elif any(x in name.lower() for x in ['cm', 'senti', 'ruas']):
        if any(spice in name.lower() for spice in ['jahe', 'kunyit', 'lengkuas', 'kencur']):
            default_weight = 5.0 * quantity  # 5g per cm untuk rempah
        else:
            default_weight = 5.0  # Default untuk ukuran cm lainnya
    else:
        default_weight = 30.0  # Default yang lebih konservatif untuk bahan lain

    logger.warning(f"Using default weight estimation for {quantity} {unit} {name}: {default_weight}g")
    return default_weight * quantity


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

        # Verifikasi berat yang tidak masuk akal
        if weight_g > 1000 and not any(
                heavy in ingredient['name'].lower() for heavy in ['daging', 'ayam', 'ikan', 'beras', 'tepung']):
            logger.warning(f"Unrealistic weight detected: {weight_g}g for {ingredient['raw']}. Attempting correction.")

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