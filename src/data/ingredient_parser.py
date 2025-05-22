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
    Ekstrak jumlah dan satuan dari teks bahan

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

    # Pattern untuk berbagai format jumlah
    # 1/2 kg, 0.5 kg, 2, 3-4, dll.
    quantity_pattern = r'^((\d+[\s-]*\d*[.,]?\d*)|(\d+\s*/\s*\d+))'
    match_qty = re.search(quantity_pattern, working_text)

    quantity = 0.0
    unit = ""
    remaining_text = working_text

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
        remaining_text = working_text[match_qty.end():].strip()

    # Extract unit if present
    common_units = [
        'kg', 'gram', 'g', 'gr', 'ons', 'liter', 'l', 'ml', 'cc', 'sdm', 'sdt',
        'butir', 'buah', 'bh', 'siung', 'bonggol', 'batang', 'lembar', 'lbr',
        'btg', 'bks', 'sachet', 'potong', 'ptg', 'iris', 'mangkok', 'gelas', 'cup',
        'bj', 'biji', 'ikat', 'papan', 'ruas', 'keping', 'keping', 'sendok', 'sloki',
        # Tambahan unit non-standar yang sering muncul
        'ekor', 'pack', 'bungkus', 'kotak', 'kuntum', 'piring'
    ]

    unit_pattern = r'^([a-zA-Z]+)'
    unit_match = re.search(unit_pattern, remaining_text)

    if unit_match:
        potential_unit = unit_match.group(1).lower()

        if 'lembar' in potential_unit or 'lbr' in potential_unit:
            unit = 'lembar'
            remaining_text = remaining_text[unit_match.end():].strip()
        elif potential_unit == 'bj' or potential_unit == 'biji':
            unit = 'bj'
            remaining_text = remaining_text[unit_match.end():].strip()
        elif potential_unit == 'papan':
            unit = 'papan'
            remaining_text = remaining_text[unit_match.end():].strip()
        else:
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

    # Handle non-standard units like "cm", "jari"
    non_standard_units = ['cm', 'jari', 'ruas', 'korek', 'api']
    for ns_unit in non_standard_units:
        if ns_unit in remaining_text.lower():
            unit = ns_unit
            remaining_text = remaining_text.replace(ns_unit, '').strip()
            break

    # Clean the remaining text as the ingredient name
    cleaned_name = normalize_ingredient_name(remaining_text)

    if '(' in cleaned_text and ')' in cleaned_text and not any(x in cleaned_name for x in INSTRUCTION_WORDS):
        parentheses_content = re.findall(r'\(([^)]*)\)', cleaned_text)
        if parentheses_content:
            clean_content = ' '.join([content for content in parentheses_content
                                      if not any(iword in content.lower() for iword in INSTRUCTION_WORDS)])
            if clean_content and clean_content not in cleaned_name:
                cleaned_name = f"{cleaned_name} {clean_content}".strip()

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
            # Only add if we have a meaningful name (not just instruction or empty)
            if (parsed_ingredient['name'] and
                    not parsed_ingredient['name'].startswith('instruction:') and
                    len(parsed_ingredient['name'].strip()) > 1):
                parsed.append(parsed_ingredient)

    # Log parsing results
    logger.info(f"Extracted {len(parsed)} ingredients from text")

    return parsed