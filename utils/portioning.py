def adjust_portion(ingredients, current_servings, target_servings):
    """Adjust ingredient amounts for a different number of servings."""
    if not current_servings or current_servings <= 0:
        current_servings = 1
    
    if not target_servings or target_servings <= 0:
        target_servings = 1
    
    # Calculate the scaling factor
    factor = target_servings / current_servings
    
    adjusted = []
    for item in ingredients:
        # Copy the item and add the adjusted amount
        adjusted_item = item.copy()
        adjusted_item["adjusted_amount"] = item["amount"] * factor
        adjusted.append(adjusted_item)
    
    return adjusted