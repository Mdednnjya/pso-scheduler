import pandas as pd

def load_data(recipe_path, nutrition_path):
    """Load recipe and nutrition data from CSV files."""
    recipes_df = pd.read_csv(recipe_path)
    nutrition_df = pd.read_csv(nutrition_path)
    return recipes_df, nutrition_df


def drop_rename_fill_and_replace(input_file, output_file, columns_to_drop, rename_mapping):
    """
    Function to drop specified columns, rename remaining columns, replace missing values with 0,
    and replace commas with periods in numeric fields.

    Args:
    input_file (str): Path to the input CSV file.
    output_file (str): Path to save the modified CSV file.
    columns_to_drop (list): List of column indices (zero-based) to drop.
    rename_mapping (dict): Dictionary mapping old column names to new column names.

    Returns:
    None
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(input_file)

        # Drop the specified columns
        df_dropped = df.drop(df.columns[columns_to_drop], axis=1)

        # Rename the remaining columns
        df_renamed = df_dropped.rename(columns=rename_mapping)

        # Replace missing values with 0
        df_filled = df_renamed.fillna(0)

        # Replace commas with periods in numeric fields
        df_cleaned = df_filled.replace(to_replace=r',', value='.', regex=True)

        # Save the modified DataFrame to a new CSV file
        df_cleaned.to_csv(output_file, index=False)
        print(f"Columns dropped, renamed, missing values replaced with 0, and commas replaced with periods! File saved at {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")