import pandas as pd
import sys
from pathlib import Path
import os
import json
import joblib
import numpy as np
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent))
from src.models.feature import FeatureEngineer

class CBFTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model_dir = 'models/'
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
        self.scaler = StandardScaler()
        os.makedirs(self.model_dir, exist_ok=True)
        
    def train(self):
        with mlflow.start_run(nested=True):
        # Load enriched recipe data
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Determine which ingredients field to use
            if 'Ingredients_Adjusted' in df.columns:
                ingredients_field = 'Ingredients_Adjusted'
            elif 'Ingredients_Enriched' in df.columns:
                ingredients_field = 'Ingredients_Enriched'
            else:
                raise ValueError("No ingredient data found in the input file")
            
            # Create text features from ingredients
            df['ingredient_text'] = df.apply(
                lambda x: ' '.join([ing.get('ingredient', '') for ing in x[ingredients_field]]), 
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
                else:
                    # Calculate from ingredients
                    total_nutrition = {
                        'calories': 0,
                        'protein': 0,
                        'fat': 0,
                        'carbohydrates': 0,
                        'fiber': 0
                    }
                    
                    for ing in row[ingredients_field]:
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
            mlflow.log_param("vocab_size", len(self.vectorizer.vocabulary_))
            
            # Normalize nutrition features
            num_features = self.scaler.fit_transform(nutrition_df)
            
            # Combine features
            feature_matrix = np.hstack((tfidf_matrix.toarray(), num_features))
            
            # Save model artifacts
            joblib.dump(self.vectorizer, os.path.join(self.model_dir, 'tfidf_vectorizer.pkl'))
            joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.pkl'))
            joblib.dump(feature_matrix, os.path.join(self.model_dir, 'feature_matrix.pkl'))
            
            mlflow.sklearn.log_model(self.vectorizer, "tfidf_vectorizer")
            mlflow.sklearn.log_model(self.scaler, "nutrition_scaler")
            
            # Save simplified recipe data for recommender
            simplified_df = df[['ID', 'Title']].copy()
            simplified_df['nutrition'] = nutrition_df.to_dict('records')
            simplified_df.to_json(os.path.join(self.model_dir, 'meal_data.json'), orient='records', indent=2)
            
            print("Model training completed!")
            return feature_matrix, simplified_df