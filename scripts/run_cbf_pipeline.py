import os
import argparse
import json
import pandas as pd
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import CBF modules
from src.models.feature import FeatureEngineer
import joblib
from sklearn.metrics.pairwise import cosine_similarity

def train_cbf_model(input_json_path, model_dir='models/'):
    """
    Train a CBF model on the provided enriched recipe data
    
    Args:
        input_json_path: Path to the enriched recipe JSON data
        model_dir: Directory to save model artifacts
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Load enriched recipe data
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} recipes from {input_json_path}")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Debug: Print a few recipe IDs to verify format
    print("Sample recipe IDs in the dataset:")
    print(df['ID'].head().tolist())
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Prepare features
    feature_matrix = feature_engineer.prepare_features(df)
    
    # Extract nutrition information
    nutrition_data = []
    for _, row in df.iterrows():
        if 'Total_Nutrition' in df.columns and not pd.isna(row['Total_Nutrition']):
            nutrition_data.append({
                'calories': row['Total_Nutrition'].get('calories', 0),
                'protein': row['Total_Nutrition'].get('protein', 0),
                'fat': row['Total_Nutrition'].get('fat', 0),
                'carbohydrates': row['Total_Nutrition'].get('carbohydrates', 0),
                'fiber': row['Total_Nutrition'].get('dietary_fiber', 0)
            })
        else:
            # Default empty nutrition for recipes without nutrition data
            nutrition_data.append({
                'calories': 0, 'protein': 0, 'fat': 0, 'carbohydrates': 0, 'fiber': 0
            })

    # Validasi nutrisi
    if sum(nutrition_data[-1].values()) == 0:
        print(f"Warning: Zero nutrition for recipe ID {row['ID']}")
        print(f"Ingredients: {row['Ingredients_Enriched']}")
        
    # Save model artifacts
    joblib.dump(feature_engineer.vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
    joblib.dump(feature_engineer.scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(feature_matrix, os.path.join(model_dir, 'feature_matrix.pkl'))
    
    # Save simplified recipe data for recommender
    simplified_df = df[['ID', 'Title']].copy()
    simplified_df['nutrition'] = nutrition_data
    
    # Convert IDs to string to ensure consistent matching
    simplified_df['ID'] = simplified_df['ID'].astype(str)
    
    # Save as JSON
    simplified_df.to_json(os.path.join(model_dir, 'meal_data.json'), orient='records', indent=2)
    
    print(f"Model trained on {len(df)} recipes with {feature_matrix.shape[1]} features")
    return feature_matrix, simplified_df

def get_recommendations(recipe_ids, n=5, max_calories=None):
    """
    Get recipe recommendations based on input recipe IDs
    
    Args:
        recipe_ids: List of recipe IDs to base recommendations on
        n: Number of recommendations to return
        max_calories: Maximum calories limit (optional)
    """
    # Check if model exists
    if not os.path.exists('models/feature_matrix.pkl'):
        print("Error: Model not found. Train the model first.")
        return None
    
    # Load model components
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_matrix = joblib.load('models/feature_matrix.pkl')
    
    # Load recipe data
    with open('models/meal_data.json', 'r', encoding='utf-8') as f:
        meal_data = json.load(f)
    
    # Convert to DataFrame
    meal_df = pd.DataFrame(meal_data)
    
    # Convert recipe IDs to string for consistent matching
    recipe_ids = [str(rid) for rid in recipe_ids]
    meal_df['ID'] = meal_df['ID'].astype(str)
    
    # Debug: Print all recipe IDs available in the dataset
    print(f"Available recipe IDs in dataset: {meal_df['ID'].head(10).tolist()}...")
    print(f"Looking for recipe IDs: {recipe_ids}")
    
    # Find indices of input recipes
    indices = meal_df[meal_df['ID'].isin(recipe_ids)].index.tolist()
    
    if not indices:
        print("Warning: No matching recipes found for the provided IDs")
        return None
    
    print(f"Found {len(indices)} matching recipes at indices: {indices}")
    
    # Get features for input recipes
    input_features = feature_matrix[indices]
    
    # Calculate similarity
    similarities = cosine_similarity(input_features, feature_matrix)
    avg_similarities = similarities.mean(axis=0)
    
    # Create similarity ranking
    sim_scores = list(enumerate(avg_similarities))
    
    # Filter by max calories if specified
    if max_calories is not None:
        valid_indices = []
        for idx, _ in sim_scores:
            recipe_calories = meal_df.iloc[idx]['nutrition']['calories']
            if recipe_calories <= max_calories:
                valid_indices.append(idx)
        
        # Filter sim_scores to only include valid indices
        sim_scores = [(idx, score) for idx, score in sim_scores if idx in valid_indices]
    
    # Sort by similarity
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get top recommendations (excluding input recipes)
    recommendations = []
    for idx, score in sim_scores:
        recipe_id = meal_df.iloc[idx]['ID']
        if recipe_id not in recipe_ids:
            recommendations.append({
                'ID': recipe_id,
                'Title': meal_df.iloc[idx]['Title'],
                'Nutrition': meal_df.iloc[idx]['nutrition'],
                'Similarity': score
            })
            if len(recommendations) >= n:
                break
    
    return recommendations

def main():
    parser = argparse.ArgumentParser(description='Recipe CBF Training and Recommendation')
    parser.add_argument('--train', action='store_true', help='Train the CBF model')
    parser.add_argument('--input', type=str, default='output/enriched_recipes.json', 
                        help='Input JSON file for training')
    parser.add_argument('--recommend', nargs='+', type=str, help='Get recommendations based on recipe IDs')
    parser.add_argument('--max-calories', type=float, help='Maximum calories for recommendations')
    parser.add_argument('--count', type=int, default=5, help='Number of recommendations to return')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    
    args = parser.parse_args()
    
    if args.train:
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} not found")
            return
        
        print(f"Training CBF model using {args.input}...")
        features, df = train_cbf_model(args.input)
        print(f"Model saved to models/ directory")
    
    if args.recommend:
        print(f"Finding recipes similar to: {args.recommend}")
        recommendations = get_recommendations(
            args.recommend,
            n=args.count,
            max_calories=args.max_calories
        )
        
        if recommendations:
            print("\nRecommendations:")
            for rec in recommendations:
                print(f"ID: {rec['ID']}")
                print(f"Title: {rec['Title']}")
                print(f"Similarity: {rec['Similarity']:.4f}")
                print("-" * 40)
            
            # Save recommendations to file
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "recommendations.json"), 'w') as f:
                json.dump(recommendations, f, indent=2)
            
            print(f"Recommendations saved to output/recommendations.json")
        else:
            print("No recommendations found. Please check if the recipe IDs exist in the dataset.")
            
            # If model exists but no recommendations, let's check what IDs are available
            if os.path.exists('models/meal_data.json'):
                with open('models/meal_data.json', 'r', encoding='utf-8') as f:
                    meal_data = json.load(f)
                    
                recipe_ids = [item['ID'] for item in meal_data]
                print(f"\nAvailable recipe IDs (first 10): {recipe_ids[:10]}")
                print(f"Total number of recipes: {len(recipe_ids)}")
                print("\nTry running the command with one of these IDs instead.")

if __name__ == "__main__":
    main()