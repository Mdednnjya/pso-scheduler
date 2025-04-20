from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import json

class FeatureEngineer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.scaler = MinMaxScaler()
    
    def fit_transform(self, text_data):
        """
        Fits and transforms text data with TF-IDF vectorizer
        """
        return self.vectorizer.fit_transform(text_data)
    
    def prepare_features(self, df):
        """
        Prepares features from the enriched recipes data format.
        This method now adapts to different data formats.
        """
        # Determine which ingredients field to use
        if 'Ingredients_Adjusted' in df.columns:
            ingredients_field = 'Ingredients_Adjusted'
        elif 'Ingredients_Enriched' in df.columns:
            ingredients_field = 'Ingredients_Enriched'
        else:
            ingredients_field = None
        
        # Create text features from ingredients if available
        if ingredients_field:
            df['ingredient_text'] = df.apply(
                lambda x: ' '.join([ing.get('ingredient', '') for ing in x[ingredients_field]]), 
                axis=1
            )
        else:
            # Fallback to Ingredients_Parsed
            df['ingredient_text'] = df.apply(
                lambda x: ' '.join(x.get('Ingredients_Parsed', [])), 
                axis=1
            )
        
        # Extract nutrition features
        nutrition_features = []
        for _, row in df.iterrows():
            # If Total_Nutrition is available, use it
            if 'Total_Nutrition' in df.columns and not pd.isna(row['Total_Nutrition']):
                total_nutrition = {
                    'calories': row['Total_Nutrition'].get('calories', 0),
                    'protein': row['Total_Nutrition'].get('protein', 0),
                    'fat': row['Total_Nutrition'].get('fat', 0),
                    'carbohydrates': row['Total_Nutrition'].get('carbohydrates', 0),
                    'fiber': row['Total_Nutrition'].get('dietary_fiber', 0)
                }
            elif ingredients_field:
                # Calculate from ingredients
                total_nutrition = {
                    'calories': 0,
                    'protein': 0,
                    'fat': 0,
                    'carbohydrates': 0,
                    'fiber': 0
                }
                
                for ing in row[ingredients_field]:
                    calories = float(ing.get('calories', 0)) if ing.get('calories') else 0
                    total_nutrition['calories'] += calories
                    
                    for nutrient in total_nutrition.keys():
                        # Handle both fiber and dietary_fiber fields
                        if nutrient == 'fiber' and 'dietary_fiber' in ing:
                            value = ing.get('dietary_fiber', 0)
                        else:
                            value = ing.get(nutrient, 0)
                        
                        try:
                            value = float(value or 0)
                            total_nutrition[nutrient] += value
                        except (ValueError, TypeError):
                            continue
            else:
                # No nutrition data available
                total_nutrition = {
                    'calories': 0,
                    'protein': 0,
                    'fat': 0,
                    'carbohydrates': 0,
                    'fiber': 0
                }
            
            nutrition_features.append([
                total_nutrition['calories'],
                total_nutrition['protein'],
                total_nutrition['fat'],
                total_nutrition['carbohydrates'],
                total_nutrition['fiber']
            ])
        
        # Create nutrition DataFrame
        nutrition_df = pd.DataFrame(
            nutrition_features,
            columns=['calories', 'protein', 'fat', 'carbohydrates', 'fiber']
        )
        
        # TF-IDF on ingredients
        tfidf_matrix = self.vectorizer.fit_transform(df['ingredient_text'])
        
        # Normalize nutrition features
        num_features = self.scaler.fit_transform(nutrition_df)
        
        # Combine features
        feature_matrix = np.hstack((tfidf_matrix.toarray(), num_features))
        return feature_matrix