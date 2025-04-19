import re

def parse_ingredient_line(line):
    # Contoh input: "100 gram udang segar"
    pattern = r"([\d.,]+)\s*(\w+)\s+(.+)"
    match = re.match(pattern, line)
    if not match:
        return None
    amount, unit, name = match.groups()
    return {
        "amount": float(amount.replace(",", ".")),
        "unit": unit.lower(),
        "ingredient": name.lower().strip()
    }
