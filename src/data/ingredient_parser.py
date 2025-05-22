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


INSTRUCTION_WORDS = [
    'iris', 'potong', 'potong2', 'cincang', 'bakar', 'rebus', 'kukus', 'goreng',
    'tumis', 'belah', 'aduk', 'campur', 'ambil', 'buang', 'haluskan',
    'tumbuk', 'geprek', 'sangrai', 'seduh', 'rendam', 'buat', 'ikat',
    'rajang', 'diiris', 'dirajang', 'digiling', 'dihaluskan', 'dipotong',
    'cuci', 'bersih', 'kupas', 'dikupas', 'sajikan', 'tambahkan', 'sesuaikan',
    'masukkan', 'masak', 'tiriskan', 'angkat', 'sisihkan', 'dinginkan', 'simpan',
    'buka', 'tutup', 'peras', 'geprek', 'siapkan', 'saring', 'tambah', 'tmbh',
    'hidangkan', 'campurkan', 'adukkan', 'sisakan', 'gulingkan', 'guling',
    'blender', 'mixer', 'ulek', 'diulek', 'diblender', 'dimixer', 'dimasak',
    'panaskan', 'dinginkan', 'bekukan', 'bekukan', 'diamkan', 'siram', 'disiram',
    'daunnya', 'saja', 'kira', 'jika', 'suka', 'bijinya', 'daging', 'isinya',
    'boleh', 'ditambah', 'sesuai', 'selera', 'kecil', 'sedang', 'besar',
    'kasar', 'halus', 'tipis', 'tebal', 'utuh', 'rendam', 'direndam', 'hancur',
    'memarkan', 'diparut', 'parut', 'sedikit', 'secukupnya', 'banyak',
    'pakai', 'gunakan', 'digunakan', 'dipakai', 'dibuat', 'buat',
    'yang', 'dan', 'atau', 'untuk', 'dengan', 'dari', 'sampai', 'hingga',
    'buang', 'dibuang', 'seruas', 'seruas', 'ruas', 'batang', 'buah',
    'sisakan', 'disajikan', 'dihidangkan', 'saji', 'hidang', 'dimakan',
    'dioles', 'oles', 'dilapisi', 'lapisi', 'direndam', 'dicampur',
    'disimpan', 'disusun', 'susun', 'diatur', 'atur', 'ditekan', 'tekan',
    'dibagi', 'bagi', 'dipotong', 'dibelah', 'belah', 'digeprek', 'dimarkan',
    'dimemarkan', 'dikocok', 'kocok', 'dishake', 'shake', 'diblender', 'blender',
    'bahan pelengkap', 'bahan lain', 'bahan', 'pelengkap', 'bumbu', 'atau',
    # Tambahan dari analisis
    'skip', 'terserah', 'boleh', 'bisa', 'pakai', 'mau', 'aja', 'juga',
    'diamkan', 'biarkan', 'suhu', 'ruang', 'min', 'max', 'jam', 'menit',
    'kocok', 'lepas', 'rata', 'olesan', 'taburan', 'pelapis', 'isian',
    'pelengkap', 'teknik', 'membentuk', 'menggoreng', 'remas'
]

COOKING_TOOLS = [
    'wajan', 'panci', 'kompor', 'teflon', 'sendok', 'blender', 'mixer',
    'rolling pin', 'spatula', 'pisau', 'talenan', 'meja', 'garpu',
    # Tambahan dari analisis
    'stik', 'bambu', 'stik bambu', 'daun pisang', 'kulit risoles',
    'kulit pangsit', 'pack', 'bungkus', 'piring', 'mangkuk', 'gelas'
]

# Blacklist untuk ingredients yang bukan makanan
NON_FOOD_BLACKLIST = [
    'marmer', 'tekel', 'meja', 'ruang', 'suhu', 'teknik', 'cara',
    'metode', 'langkah', 'tahap', 'proses', 'bagian', 'porsi'
]

# Kata-kata yang menandakan ini bukan ingredient utama
SKIP_INDICATORS = [
    'skip', 'nggak punya', 'tidak ada', 'kosong', 'habis', 'terserah',
    'boleh', 'bisa', 'optional', 'pilihan', 'sesuai selera'
]


def is_non_food_item(name):
    """
    Check if the parsed name is a non-food item (cooking tool, instruction, etc.)
    """
    name_lower = name.lower()

    # Check for cooking tools
    for tool in COOKING_TOOLS:
        if tool in name_lower:
            return True

    # Check for non-food blacklist
    for item in NON_FOOD_BLACKLIST:
        if item in name_lower:
            return True

    # Check for skip indicators
    for indicator in SKIP_INDICATORS:
        if indicator in name_lower:
            return True

    # Check if it's just numbers
    if re.match(r'^\d+(\s+\d+)*$', name_lower.strip()):
        return True

    # Check if it's measurement without ingredient
    measurement_only = re.match(r'^(\d+\s*)?(cm|mm|jari|korek|api|setengah|bagian)\s*$', name_lower.strip())
    if measurement_only:
        return True

    return False


def normalize_ingredient_name(name):
    """
    Normalisasi nama bahan dengan menghapus kata-kata yang tidak relevan
    """
    if not name:
        return ""

    # Lowercase
    normalized = name.lower()

    # Check if this is a non-food item first
    if is_non_food_item(normalized):
        return "instruction:" + normalized

    # Kata-kata yang biasanya tidak relevan untuk identifikasi bahan
    words_to_remove = [
        'segar', 'rebus', 'mentah', 'goreng', 'kering', 'matang',
        'iris', 'potong', 'potong2', 'cincang', 'halus', 'parut', 'geprek',
        'secukupnya', 'sesuai selera', 'sedang', 'besar', 'kecil',
        'diiris', 'dikupas', 'dipotong', 'dihaluskan', 'dirajang', 'digiling',
        'cuci', 'bersih', 'kupas', 'utuh', 'sesuai', 'selera', 'yang', 'sudah',
        'dan', 'atau', 'dengan', 'untuk', 'di', 'dari', 'ke', 'pada', 'jadi',
        'saya', 'pakai', 'pake', 'aku', 'ku', 'kamu', 'tambah', 'tmbh',
        'suka', 'yg', 'dg', 'lg', 'jg', 'nya', 'nya:', 'kotak2', 'kecil', 'cukup',
        'sampai', 'skip', 'nggak', 'punya', 'terserah', 'aja', 'sj', 'mau',
        'boleh', 'bisa', 'juga'
    ]

    # Cek apakah ini instruksi, bukan bahan
    for word in INSTRUCTION_WORDS:
        if normalized == word or normalized.startswith(word + ' '):
            return "instruction:" + normalized

    # Hapus kata-kata tidak relevan
    for word in words_to_remove:
        normalized = re.sub(r'\b' + word + r'\b', '', normalized)

    # Hapus karakter non-alfabet dan whitespace berlebih
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    # Filter hasil yang terlalu pendek atau tidak bermakna
    if len(normalized) < 2 or normalized.isdigit():
        return "instruction:" + name.lower()

    # Standardisasi beberapa nama khusus
    name_mappings = {
        'cabe': 'cabai',
        'bj cabai': 'cabai',
        'bj cabe': 'cabai',
        'cabai rawit': 'cabai rawit',
        'cabe rawit': 'cabai rawit',
        'cabai merah': 'cabai merah',
        'cabe merah': 'cabai merah',
        'bw merah': 'bawang merah',
        'bw putih': 'bawang putih',
        'bamer': 'bawang merah',
        'baput': 'bawang putih',
        'saos': 'saus',
        'sauce': 'saus',
        'baby jagung': 'baby corn',
        'baby corn': 'baby corn',
        'royko': 'kaldu bubuk',
        'masako': 'kaldu bubuk',
        'maggi': 'kaldu bubuk',
        'penyedap': 'kaldu bubuk',
        'micin': 'penyedap rasa',
        # Tambahan dari analisis
        'maizena': 'tepung maizena',
        'meizena': 'tepung maizena',
        'beras basmathi': 'beras basmati',
        'lada hitam': 'merica hitam',
        'jintan': 'biji jintan',
        'telor': 'telur',
        'timun': 'ketimun',
        'klapa': 'kelapa',
        'me yes': 'mayonaise',
        'mayonaise': 'mayonaise',
        'sckp': 'secukupnya'
    }

    # Terapkan pemetaan nama
    for old, new in name_mappings.items():
        if normalized == old or normalized.startswith(old + ' '):
            normalized = normalized.replace(old, new, 1)
            break

    return normalized


def extract_quantity_and_unit(ingredient_text):
    """
    Enhanced extraction with better unit parsing and validation

    Returns:
        tuple: (quantity, unit, cleaned_name)
    """
    if not ingredient_text:
        return 0.0, "", ""

    # Clean text
    cleaned_text = clean_text(ingredient_text)

    # Hapus teks dalam kurung
    cleaned_text_no_parentheses = re.sub(r'\([^)]*\)', '', cleaned_text).strip()

    # Gunakan teks tanpa kurung untuk ekstraksi quantity dan unit
    working_text = cleaned_text_no_parentheses if cleaned_text_no_parentheses else cleaned_text

    # Enhanced pattern untuk berbagai format jumlah
    # Handle: "250 gr", "2 gram", "10 butir", "1/2 kg", "3-4 siung"
    quantity_pattern = r'^(\d+(?:[.,]\d+)?(?:\s*/\s*\d+(?:[.,]\d+)?)?(?:\s*-\s*\d+(?:[.,]\d+)?)?)'
    match_qty = re.search(quantity_pattern, working_text)

    quantity = 0.0
    unit = ""
    remaining_text = working_text

    if match_qty:
        qty_str = match_qty.group(1).strip()

        # Handle fractions (e.g. "1/2")
        if '/' in qty_str:
            try:
                num, denom = qty_str.split('/')
                quantity = float(num.strip()) / float(denom.strip())
            except (ValueError, ZeroDivisionError):
                quantity = 1.0
        # Handle ranges (e.g. "3-4")
        elif '-' in qty_str:
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
        remaining_text = working_text[match_qty.end():].strip()

    # Enhanced unit extraction with validation
    # Priority order: exact match > partial match > default
    unit_patterns = {
        # Weight units (high priority)
        r'\b(kg|kilogram)\b': 'kg',
        r'\b(gr?|gram)\b': 'g',  # Fix: "gr" should be "g", not "api"
        r'\b(ons)\b': 'ons',

        # Volume units
        r'\b(liter|l)\b': 'liter',
        r'\b(ml|mililiter)\b': 'ml',
        r'\b(cc)\b': 'cc',

        # Cooking units
        r'\b(sdm|sendok\s+makan)\b': 'sdm',
        r'\b(sdt|sendok\s+teh)\b': 'sdt',
        r'\b(gelas)\b': 'gelas',
        r'\b(mangkok)\b': 'mangkok',
        r'\b(cup)\b': 'cup',

        # Count units (validate with ingredient type)
        r'\b(butir|btr)\b': 'butir',
        r'\b(buah|bh)\b': 'buah',
        r'\b(siung)\b': 'siung',
        r'\b(lembar|lbr)\b': 'lembar',
        r'\b(batang|btg)\b': 'batang',
        r'\b(ruas)\b': 'ruas',
        r'\b(biji|bj)\b': 'biji',
        r'\b(ekor)\b': 'ekor',
        r'\b(potong|ptg)\b': 'potong',
        r'\b(iris)\b': 'iris',
        r'\b(keping)\b': 'keping',
        r'\b(papan)\b': 'papan',
        r'\b(ikat|ikt)\b': 'ikat',
        r'\b(pack|bungkus|bks)\b': 'pack',
        r'\b(sachet)\b': 'sachet',

        # Special cases
        r'\b(secukupnya)\b': 'secukupnya'
    }

    # Try to match unit patterns
    for pattern, standard_unit in unit_patterns.items():
        match = re.search(pattern, remaining_text, re.IGNORECASE)
        if match:
            unit = standard_unit
            # Remove matched unit from remaining text
            remaining_text = re.sub(pattern, '', remaining_text, flags=re.IGNORECASE).strip()
            break

    # Validate unit-ingredient compatibility
    unit, quantity = validate_unit_ingredient_compatibility(unit, quantity, remaining_text)

    # Clean the remaining text as the ingredient name
    cleaned_name = normalize_ingredient_name(remaining_text)

    return quantity, unit, cleaned_name


def validate_unit_ingredient_compatibility(unit, quantity, ingredient_name):
    """
    Validate and correct unit-ingredient combinations

    Args:
        unit: Detected unit
        quantity: Detected quantity
        ingredient_name: Ingredient name

    Returns:
        tuple: (corrected_unit, corrected_quantity)
    """
    name_lower = ingredient_name.lower()

    # Specific validation rules
    validations = {
        # Telur validation
        'telur': {
            'valid_units': ['butir', 'biji', 'buah', ''],
            'invalid_units': ['kg', 'g', 'gr'],
            'default_unit': 'butir',
            'realistic_range': (1, 20)
        },
        'telor': {
            'valid_units': ['butir', 'biji', 'buah', ''],
            'invalid_units': ['kg', 'g', 'gr'],
            'default_unit': 'butir',
            'realistic_range': (1, 20)
        },

        # Merica/Lada validation
        'merica': {
            'valid_units': ['sdt', 'butir', 'biji', 'g', 'gr'],
            'invalid_units': ['kg', 'buah', 'ekor'],
            'default_unit': 'sdt',
            'realistic_range': (0.25, 10)  # 1/4 sdt to 2 sdt max
        },
        'lada': {
            'valid_units': ['sdt', 'butir', 'biji', 'g', 'gr'],
            'invalid_units': ['kg', 'buah', 'ekor'],
            'default_unit': 'sdt',
            'realistic_range': (0.25, 10)
        },

        # Bawang validation
        'bawang': {
            'valid_units': ['siung', 'buah', 'g', 'gr', 'kg'],
            'invalid_units': ['butir', 'lembar'],
            'default_unit': 'siung',
            'realistic_range': (1, 20)
        },

        # Daging validation
        'daging': {
            'valid_units': ['g', 'gr', 'kg', 'potong'],
            'invalid_units': ['butir', 'siung', 'lembar'],
            'default_unit': 'g',
            'realistic_range': (50, 2000)  # 50g to 2kg
        },
        'sapi': {
            'valid_units': ['g', 'gr', 'kg', 'potong'],
            'invalid_units': ['butir', 'siung', 'lembar'],
            'default_unit': 'g',
            'realistic_range': (100, 2000)
        },

        # Udang validation
        'udang': {
            'valid_units': ['g', 'gr', 'kg', 'ekor'],
            'invalid_units': ['butir', 'siung', 'lembar'],
            'default_unit': 'g',
            'realistic_range': (50, 1000)  # 50g to 1kg
        },

        # Daun validation
        'daun': {
            'valid_units': ['lembar', 'lbr', 'ikat'],
            'invalid_units': ['kg', 'g', 'butir'],
            'default_unit': 'lembar',
            'realistic_range': (1, 50)
        }
    }

    # Check if ingredient matches any validation rules
    for ingredient_key, rules in validations.items():
        if ingredient_key in name_lower:
            # Check if current unit is invalid
            if unit in rules['invalid_units']:
                logger.warning(
                    f"Invalid unit '{unit}' for '{ingredient_name}', using default '{rules['default_unit']}'")
                unit = rules['default_unit']

            # Check quantity range
            min_qty, max_qty = rules['realistic_range']
            if quantity < min_qty or quantity > max_qty:
                # Special case: if quantity is way too high, might be weight issue
                if quantity > max_qty * 10 and unit in ['g', 'gr']:
                    # Possible gram/kilogram confusion
                    corrected_qty = quantity / 1000
                    if min_qty <= corrected_qty <= max_qty:
                        logger.warning(
                            f"Quantity {quantity}g seems too high for '{ingredient_name}', correcting to {corrected_qty}kg")
                        quantity = corrected_qty
                        unit = 'kg'
                elif quantity < min_qty and unit in ['g', 'gr'] and ingredient_key in ['daging', 'udang']:
                    # Likely missing digits (2g udang -> 200g udang)
                    corrected_qty = quantity * 100
                    if min_qty <= corrected_qty <= max_qty:
                        logger.warning(
                            f"Quantity {quantity}g too low for '{ingredient_name}', correcting to {corrected_qty}g")
                        quantity = corrected_qty
            break

    return unit, quantity


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

    # Enhanced separators - handle "--" specifically for Kaggle dataset
    # Priority: "--" > "\n" > "-" > "•" > "*" > ","
    lines = []

    # First, split by "--" (Kaggle dataset format)
    if '--' in ingredient_text:
        parts = ingredient_text.split('--')
        lines.extend([part.strip() for part in parts if part.strip()])
    else:
        # Fall back to other separators
        parts = re.split(r'\n|-|•|\*|,', ingredient_text)
        lines.extend([part.strip() for part in parts if part.strip()])

    # Parse each line
    parsed = []
    for line in lines:
        if line and not line.isspace():
            # Skip obvious non-ingredient lines
            skip_patterns = [
                r'^(bahan|bumbu|pelengkap|utama|lain|lainnya)\s*:?\s*$',
                r'^(untuk|sebagai|cara|langkah)\s',
                r'^\s*-+\s*$'
            ]

            if any(re.match(pattern, line.lower()) for pattern in skip_patterns):
                continue

            parsed_ingredient = parse_ingredient_line(line.strip())

            # Only add if we have a meaningful name and it's not an instruction
            if (parsed_ingredient['name'] and
                    not parsed_ingredient['name'].startswith('instruction:') and
                    len(parsed_ingredient['name'].strip()) > 1):
                parsed.append(parsed_ingredient)

    # Log parsing results
    logger.info(f"Extracted {len(parsed)} ingredients from text: {[p['name'] for p in parsed]}")

    return parsed