import re


def clean_ingredient_name(name):
    """Clean ingredient name but preserve key information."""
    # Normalisasi spasi dan karakter khusus
    name = name.lower().strip()
    name = re.sub(r'\s+', ' ', name)

    # Hapus simbol emoji dan karakter khusus
    name = re.sub(r'[^\w\s.,()]', '', name)

    return name.strip()


def parse_ingredient_line(line):
    """Parse a single ingredient line to extract the ingredient name."""
    # Bersihkan line terlebih dahulu
    line = clean_ingredient_name(line)

    # Hilangkan angka dan satuan secara lebih hati-hati
    line = re.sub(r'\d+[\/\.,]*\d*\s*', '', line)
    line = re.sub(
        r'\b(kg|gram|gr|g|sdm|sdt|butir|buah|siung|ikat|batang|lembar|mangkok|sendok|ml|cc|ons|bungkus|sachet)\b', '',
        line)

    # Hapus kata-kata instruksi umum
    common_instructions = [
        'secukupnya', 'sedikit', 'sesuai selera', 'iris', 'potong', 'cincang',
        'haluskan', 'geprek', 'parut', 'diparut', 'diiris', 'dipotong'
    ]

    for instruction in common_instructions:
        line = re.sub(r'\b' + instruction + r'\b', '', line)

    # Bersihkan spasi berlebih lagi
    line = re.sub(r'\s+', ' ', line)

    return line.strip()


def extract_ingredients(ingredient_text):
    """Extract individual ingredients from a full ingredient text."""
    if not ingredient_text or isinstance(ingredient_text, float):
        return []

    # Split teks bahan berdasarkan pemisah umum
    lines = re.split(r'--|\n|-|•|\*|,', ingredient_text)

    # Parse setiap baris dan filter yang kosong
    parsed = []
    for line in lines:
        if line and not line.isspace():
            parsed_line = parse_ingredient_line(line)
            if parsed_line and not parsed_line.isspace():
                parsed.append(parsed_line)

    # Hapus duplikat dan kembalikan
    return list(set(parsed))