import re

def parse_ingredient_line(line):
    """Parse an ingredient line to extract amount, unit, and name."""
    # Pattern for "[amount] [unit] [ingredient]"
    pattern = r"([\d.,]+)\s*(\w+)\s+(.+)"
    match = re.match(pattern, line.strip())
    
    if match:
        amount_str, unit, name = match.groups()
        # Convert amount to float, replacing commas with dots
        try:
            amount = float(amount_str.replace(",", "."))
        except ValueError:
            amount = 0
            
        return {
            "amount": amount,
            "unit": unit.lower(),
            "ingredient": name.lower().strip()
        }
    else:
        # Fallback for lines that don't match the pattern
        return {
            "amount": 100,  # Default amount
            "unit": "gram",  # Default unit
            "ingredient": line.lower().strip()
        }