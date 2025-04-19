def adjust_portion(ingredients, current_servings, target_servings):
    factor = target_servings / current_servings
    adjusted = []
    for item in ingredients:
        adjusted.append({
            **item,
            "adjusted_amount": item["amount"] * factor
        })
    return adjusted
