# src/models/feature.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    def __init__(self):
        """Initialize the feature engineer"""
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.scaler = StandardScaler()

    def prepare_features(self, recipe_df):
        """
        Prepare features for content-based filtering

        Args:
            recipe_df: DataFrame containing recipes with ingredients and nutrition

        Returns:
            Feature matrix combining text features and nutrition features
        """
        # Extract ingredients as text
        if 'Ingredients_Parsed' in recipe_df.columns:
            ingredients_text = recipe_df['Ingredients_Parsed'].apply(
                lambda x: ' '.join(x) if isinstance(x, list) else ''
            )
        else:
            # Fall back to recipe titles if ingredients not available
            ingredients_text = recipe_df['Title']

        # Create text features using TF-IDF
        text_features = self.vectorizer.fit_transform(ingredients_text).toarray()

        # Extract nutrition features if available
        if 'nutrition' in recipe_df.columns:
            # Extract nutrition values into separate columns
            nutrition_features = np.array([
                [
                    row['nutrition'].get('calories', 0),
                    row['nutrition'].get('protein', 0),
                    row['nutrition'].get('fat', 0),
                    row['nutrition'].get('carbohydrates', 0),
                    row['nutrition'].get('fiber', 0)
                ]
                for _, row in recipe_df.iterrows()
            ])

            # Scale nutrition features
            nutrition_features = self.scaler.fit_transform(nutrition_features)

            # Combine text and nutrition features
            combined_features = np.hstack([text_features, nutrition_features])
        else:
            combined_features = text_features

        return combined_features