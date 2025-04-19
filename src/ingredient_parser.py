import re

def parse_ingredient_line(line):
    """Parse a single ingredient line to extract the ingredient name."""
    line = line.lower()
    # Hilangkan angka dan satuan
    line = re.sub(r'[\d/.,]+', '', line)
    line = re.sub(r'\b(kg|gram|gr|sdm|sdt|butir|buah|siung|ikat|batang|lembar|mangkok|sendok|ml|cc)\b', '', line)
    # Hilangkan karakter non-alfabet
    line = re.sub(r'[^a-zA-Z\s]', '', line)
    # Normalisasi dan strip whitespace
    return line.strip()

def extract_ingredients(ingredient_text):
    """Extract individual ingredients from a full ingredient text."""
    lines = re.split(r'--|\n|-', ingredient_text)
    parsed = [parse_ingredient_line(line) for line in lines if line.strip()]
    return list(set(parsed))  # Unikkan