import re
import unicodedata
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ingredient_parser')


def clean_text(text):
    """
    Membersihkan teks dari karakter khusus dan normalisasi
    """
    if not text or not isinstance(text, str):
        return ""

    # Konversi fractions ke desimal
    text = text.replace('½', '0.5').replace('¼', '0.25').replace('¾', '0.75')
    text = text.replace('⅓', '0.33').replace('⅔', '0.67')

    # Normalisasi unicode
    text = unicodedata.normalize('NFKD', text)

    # Hapus karakter non-alfanumerik kecuali spasi
    text = re.sub(r'[^\w\s.,()]', ' ', text)

    # Normalisasi spasi
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def normalize_ingredient_name(name):
    """
    Normalisasi nama bahan dengan menghapus kata-kata yang tidak relevan
    """
    if not name:
        return ""

    # Lowercase
    normalized = name.lower()

    # Kata-kata yang biasanya tidak relevan untuk identifikasi bahan
    words_to_remove = [
        'segar', 'rebus', 'mentah', 'goreng', 'kering', 'matang',
        'iris', 'potong', 'cincang', 'halus', 'parut', 'geprek',
        'secukupnya', 'sesuai selera', 'sedang', 'besar', 'kecil'
    ]

    for word in words_to_remove:
        normalized = re.sub(r'\b' + word + r'\b', '', normalized)

    # Hapus karakter non-alfabet dan whitespace berlebih
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    return normalized


def extract_quantity_and_unit(ingredient_text):
    """
    Ekstrak jumlah dan satuan dari teks bahan

    Returns:
        tuple: (quantity, unit, cleaned_name)
    """
    if not ingredient_text:
        return 0.0, "", ""

    # Clean text
    cleaned_text = clean_text(ingredient_text)

    # Pattern untuk berbagai format jumlah
    # 1/2 kg, 0.5 kg, 2, 3-4, dll.
    quantity_pattern = r'^((\d+[\s-]*\d*[.,]?\d*)|(\d+\s*/\s*\d+))'
    match_qty = re.search(quantity_pattern, cleaned_text)

    quantity = 0.0
    unit = ""
    remaining_text = cleaned_text

    if match_qty:
        qty_str = match_qty.group(1)

        # Handle fractions (e.g. "1/2")
        if '/' in qty_str:
            try:
                num, denom = qty_str.split('/')
                quantity = float(num.strip()) / float(denom.strip())
            except (ValueError, ZeroDivisionError):
                quantity = 1.0
        else:
            # Handle ranges (e.g. "3-4")
            if '-' in qty_str:
                try:
                    low, high = qty_str.split('-')
                    quantity = (float(low.strip()) + float(high.strip())) / 2
                except ValueError:
                    quantity = 1.0
            else:
                # Handle simple numbers
                try:
                    quantity = float(qty_str.replace(',', '.').strip())
                except ValueError:
                    quantity = 1.0

        # Remove quantity from text for further processing
        remaining_text = cleaned_text[match_qty.end():].strip()

    # Extract unit if present
    common_units = [
        'kg', 'gram', 'g', 'gr', 'ons', 'liter', 'l', 'ml', 'cc', 'sdm', 'sdt',
        'butir', 'buah', 'bh', 'siung', 'bonggol', 'batang', 'lembar', 'lbr',
        'btg', 'bks', 'sachet', 'potong', 'ptg', 'iris', 'mangkok', 'gelas', 'cup'
    ]

    unit_pattern = r'^([a-zA-Z]+)'
    unit_match = re.search(unit_pattern, remaining_text)

    if unit_match:
        potential_unit = unit_match.group(1).lower()

        # Check if the potential unit is in our list of common units
        for common_unit in common_units:
            if potential_unit == common_unit or potential_unit.startswith(common_unit):
                unit = common_unit
                # Remove unit from text
                remaining_text = remaining_text[unit_match.end():].strip()
                break

    # Handle special case for "secukupnya"
    if 'secukupnya' in cleaned_text.lower() or 'sesuai selera' in cleaned_text.lower():
        if quantity == 0.0:
            quantity = 1.0
        unit = 'secukupnya'

    # Clean the remaining text as the ingredient name
    cleaned_name = normalize_ingredient_name(remaining_text)

    return quantity, unit, cleaned_name


def parse_ingredient_line(line):
    """
    Parse a single ingredient line to extract quantity, unit, and name

    Returns:
        dict: Dictionary with quantity, unit, and name
    """
    quantity, unit, name = extract_quantity_and_unit(line)

    return {
        'raw': line,
        'quantity': quantity,
        'unit': unit,
        'name': name
    }


def extract_ingredients(ingredient_text):
    """
    Extract individual ingredients from a full ingredient text

    Returns:
        list: List of parsed ingredient dictionaries
    """
    if not ingredient_text or isinstance(ingredient_text, float):
        return []

    # Split by common separators
    lines = re.split(r'--|\n|-|•|\*|,', ingredient_text)

    # Parse each line
    parsed = []
    for line in lines:
        if line and not line.isspace():
            parsed_ingredient = parse_ingredient_line(line.strip())
            if parsed_ingredient['name']:  # Only add if we have a name
                parsed.append(parsed_ingredient)

    # Log parsing results
    logger.info(f"Extracted {len(parsed)} ingredients from text")

    return parsed