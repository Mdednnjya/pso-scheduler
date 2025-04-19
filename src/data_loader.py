import pandas as pd

def load_data(recipe_path, nutrition_path):
    """Load recipe and nutrition data from CSV files."""
    recipes_df = pd.read_csv(recipe_path)
    nutrition_df = pd.read_csv(nutrition_path)
    return recipes_df, nutrition_df