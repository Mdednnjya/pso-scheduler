import joblib
import numpy as np
import pandas as pd
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from src.models.user_preferences import UserPreferences


class CBFRecommender:
    def __init__(self, model_dir='models/'):
        self.model_dir = model_dir
        self.load_models()

    def load_models(self):
        """Load all model artifacts"""
        model_files = [
            'tfidf_vectorizer.pkl',
            'scaler.pkl',
            'feature_matrix.pkl',
            'meal_data.json'
        ]

        # Check if all model files exist
        for file in model_files:
            if not os.path.exists(os.path.join(self.model_dir, file)):
                raise FileNotFoundError(
                    f"Model file {file} not found in {self.model_dir}. Please train the model first.")

        self.vectorizer = joblib.load(os.path.join(self.model_dir, 'tfidf_vectorizer.pkl'))
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
        self.feature_matrix = joblib.load(os.path.join(self.model_dir, 'feature_matrix.pkl'))

        # Load recipe data
        with open(os.path.join(self.model_dir, 'meal_data.json'), 'r', encoding='utf-8') as f:
            self.meal_data = pd.DataFrame(json.load(f))

        # Ensure IDs are strings for consistent matching
        self.meal_data['ID'] = self.meal_data['ID'].astype(str)

    def recommend(self, recipe_ids, n=5, max_calories=None, user_preferences=None):
        """
        Recommend recipes based on similarity to input recipe IDs and user preferences

        Args:
            recipe_ids: List of recipe IDs to base recommendations on
            n: Number of recommendations to return
            max_calories: Maximum calories per recommendation (optional filter)
            user_preferences: UserPreferences object for filtering recipes

        Returns:
            DataFrame with recommended recipes
        """
        # Convert all recipe IDs to strings for consistent matching
        recipe_ids = [str(rid) for rid in recipe_ids]

        # Apply user preferences filtering if provided
        if user_preferences:
            # Filter the meal data based on user preferences
            filtered_meal_data = user_preferences.filter_recipes(self.meal_data)

            if filtered_meal_data.empty:
                print("Warning: No recipes match the user preferences.")
                return pd.DataFrame()

            # Get the indices of the filtered recipes
            filtered_indices = filtered_meal_data.index.tolist()

            # Create a filtered feature matrix
            filtered_feature_matrix = self.feature_matrix[filtered_indices]
        else:
            filtered_meal_data = self.meal_data
            filtered_feature_matrix = self.feature_matrix
            filtered_indices = self.meal_data.index.tolist()

        # Find indices of input recipes
        indices = filtered_meal_data[filtered_meal_data['ID'].isin(recipe_ids)].index.tolist()

        if not indices:
            # Try looking in the original meal data to see if the input recipes were filtered out
            original_indices = self.meal_data[self.meal_data['ID'].isin(recipe_ids)].index.tolist()
            if original_indices:
                print("Warning: Input recipes don't meet the preference criteria and were filtered out.")
            else:
                print(f"Warning: No matching recipes found for the provided IDs: {recipe_ids}")
                print(f"Available IDs (first 5): {filtered_meal_data['ID'].head().tolist()}")
            return pd.DataFrame()

        # Get features for input recipes
        input_features = self.feature_matrix[indices]

        # Calculate similarity between input recipes and filtered recipes
        similarities = cosine_similarity(input_features, filtered_feature_matrix)
        avg_similarities = similarities.mean(axis=0)

        # Create similarity ranking (using the indices from the filtered dataset)
        sim_scores = list(enumerate(avg_similarities))

        # Filter by max calories if specified
        if max_calories is not None:
            valid_indices = []
            for idx, _ in sim_scores:
                recipe_calories = float(filtered_meal_data.iloc[idx]['nutrition']['calories'])
                if recipe_calories <= max_calories:
                    valid_indices.append(idx)

            # Filter sim_scores to only include valid indices
            sim_scores = [(idx, score) for idx, score in sim_scores if idx in valid_indices]

        # Sort by similarity
        sim_scores.sort(key=lambda x: x[1], reverse=True)

        # Get top recommendations (excluding input recipes)
        recommendations = []
        for idx, score in sim_scores:
            recipe_id = filtered_meal_data.iloc[idx]['ID']
            if recipe_id not in recipe_ids:
                recommendations.append(recipe_id)
                if len(recommendations) >= n:
                    break

        # Return recommended recipes
        result_df = filtered_meal_data[filtered_meal_data['ID'].isin(recommendations)].copy()

        # Add similarity scores
        sim_dict = {filtered_meal_data.iloc[idx]['ID']: score for idx, score in sim_scores}
        result_df['similarity'] = result_df['ID'].map(sim_dict)

        # Sort by similarity
        result_df = result_df.sort_values(by='similarity', ascending=False)

        return result_df