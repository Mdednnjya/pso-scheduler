import os
import argparse
import json
import pandas as pd
import sys
from pathlib import Path
import numpy as np
import mlflow

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import CBF modules
from src.models.cbf_trainer import CBFTrainer
from src.models.feature import FeatureEngineer
from src.models.user_preferences import UserPreferences
import joblib
from sklearn.metrics.pairwise import cosine_similarity


def normalize_nutrition_values(nutrition_data, servings=2):
    """
    Normalize nutrition values by dividing by the number of servings

    Args:
        nutrition_data: List of nutrition dictionaries
        servings: Number of servings to assume (default: 4)

    Returns:
        List of normalized nutrition dictionaries
    """
    normalized_data = []
    for item in nutrition_data:
        normalized_item = {
            'calories': round(item['calories'] / servings, 1),
            'protein': round(item['protein'] / servings, 1),
            'fat': round(item['fat'] / servings, 1),
            'carbohydrates': round(item['carbohydrates'] / servings, 1),
            'fiber': round(item['fiber'] / servings, 1)
        }
        normalized_data.append(normalized_item)

    return normalized_data


def train_cbf_model(input_json_path, model_dir='models/', servings=2):
    """
    Train a CBF model on the provided enriched recipe data

    Args:
        input_json_path: Path to the enriched recipe JSON data
        model_dir: Directory to save model artifacts
        servings: Number of servings to normalize nutrition values by
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
        elif 'Ingredients_Enriched' in df.columns:
            # Calculate total nutrition from enriched ingredients
            total_calories = sum(float(ing.get('calories', 0) or 0) for ing in row['Ingredients_Enriched'])
            total_protein = sum(float(ing.get('protein', 0) or 0) for ing in row['Ingredients_Enriched'])
            total_fat = sum(float(ing.get('fat', 0) or 0) for ing in row['Ingredients_Enriched'])
            total_carbs = sum(float(ing.get('carbohydrates', 0) or 0) for ing in row['Ingredients_Enriched'])
            total_fiber = sum(float(ing.get('fiber', 0) or 0) for ing in row['Ingredients_Enriched'])

            nutrition_data.append({
                'calories': total_calories,
                'protein': total_protein,
                'fat': total_fat,
                'carbohydrates': total_carbs,
                'fiber': total_fiber
            })
        else:
            # Default empty nutrition for recipes without nutrition data
            nutrition_data.append({
                'calories': 0, 'protein': 0, 'fat': 0, 'carbohydrates': 0, 'fiber': 0
            })

    # Normalize nutrition data by servings
    print(f"Normalizing nutrition values by {servings} servings...")
    normalized_nutrition = normalize_nutrition_values(nutrition_data, servings)

    # Print a sample of before/after normalization for verification
    if len(nutrition_data) > 0:
        print("\nNutrition normalization example:")
        print(f"Original: Calories: {nutrition_data[0]['calories']}, Protein: {nutrition_data[0]['protein']}g")
        print(
            f"Normalized: Calories: {normalized_nutrition[0]['calories']}, Protein: {normalized_nutrition[0]['protein']}g")

    # Validasi nutrisi
    for i, (orig, norm) in enumerate(zip(nutrition_data, normalized_nutrition)):
        if sum(orig.values()) == 0:
            print(f"Warning: Zero nutrition for recipe ID {df.iloc[i]['ID']}")
            if 'Ingredients_Enriched' in df.columns:
                print(f"Ingredients: {df.iloc[i]['Ingredients_Enriched']}")

    # Save model artifacts
    joblib.dump(feature_engineer.vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
    joblib.dump(feature_engineer.scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(feature_matrix, os.path.join(model_dir, 'feature_matrix.pkl'))

    # Save simplified recipe data for recommender
    simplified_df = df[['ID', 'Title']].copy()
    simplified_df['nutrition'] = normalized_nutrition  # Use normalized nutrition

    # Save ingredients information for preference filtering
    if 'Ingredients_Parsed' in df.columns:
        simplified_df['Ingredients_Parsed'] = df['Ingredients_Parsed']
    if 'Ingredients_Enriched' in df.columns:
        simplified_df['Ingredients_Enriched'] = df['Ingredients_Enriched']

    # Convert IDs to string to ensure consistent matching
    simplified_df['ID'] = simplified_df['ID'].astype(str)

    # Save as JSON
    simplified_df.to_json(os.path.join(model_dir, 'meal_data.json'), orient='records', indent=2)

    print(f"Model trained on {len(df)} recipes with {feature_matrix.shape[1]} features")
    print(f"Normalized nutrition data saved to {os.path.join(model_dir, 'meal_data.json')}")
    return feature_matrix, simplified_df


def get_recommendations(recipe_ids, n=5, max_calories=None, excluded_ingredients=None, dietary_type=None):
    """
    Get recipe recommendations based on input recipe IDs and user preferences

    Args:
        recipe_ids: List of recipe IDs to base recommendations on
        n: Number of recommendations to return
        max_calories: Maximum calories limit (optional)
        excluded_ingredients: List of ingredients to exclude
        dietary_type: Dietary preference (vegetarian, vegan, pescatarian)
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

    # Create user preferences object if needed
    user_preferences = None
    if excluded_ingredients or dietary_type:
        user_preferences = UserPreferences(
            excluded_ingredients=excluded_ingredients,
            dietary_type=dietary_type
        )

        # Apply filtering
        filtered_meal_df = user_preferences.filter_recipes(meal_df)

        if filtered_meal_df.empty:
            print("Warning: No recipes match the user preferences.")
            return None

        print(f"Filtered from {len(meal_df)} to {len(filtered_meal_df)} recipes based on preferences")
        meal_df = filtered_meal_df

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
            try:
                recipe_calories = float(meal_df.iloc[idx]['nutrition']['calories'])
                if recipe_calories <= max_calories:
                    valid_indices.append(idx)
            except (IndexError, KeyError):
                continue

        # Filter sim_scores to only include valid indices
        sim_scores = [(idx, score) for idx, score in sim_scores if idx in valid_indices]

    # Sort by similarity
    sim_scores.sort(key=lambda x: x[1], reverse=True)

    # Get top recommendations (excluding input recipes)
    recommendations = []
    for idx, score in sim_scores:
        try:
            recipe_id = meal_df.iloc[idx]['ID']
            if recipe_id not in recipe_ids:
                recommendations.append({
                    'ID': recipe_id,
                    'Title': meal_df.iloc[idx]['Title'],
                    'Nutrition': meal_df.iloc[idx]['nutrition'],
                    'Similarity': float(score)  # Convert numpy float to Python float for JSON serialization
                })
                if len(recommendations) >= n:
                    break
        except IndexError:
            continue

    return recommendations


def main():
    parser = argparse.ArgumentParser(description='Recipe CBF Training and Recommendation with User Preferences')
    parser.add_argument('--train', action='store_true', help='Train the CBF model')
    parser.add_argument('--input', type=str, default='output/enriched_recipes.json',
                        help='Input JSON file for training')
    parser.add_argument('--recommend', nargs='+', type=str, help='Get recommendations based on recipe IDs')
    parser.add_argument('--max-calories', type=float, help='Maximum calories for recommendations')
    parser.add_argument('--count', type=int, default=5, help='Number of recommendations to return')
    parser.add_argument('--exclude', nargs='+', help='Ingredients to exclude (e.g., chicken beef)')
    parser.add_argument('--diet-type', choices=['vegetarian', 'vegan', 'pescatarian'],
                        help='Dietary preference')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    parser.add_argument('--servings', type=int, default=4,
                        help='Number of servings to normalize nutrition values by (default: 4)')
    parser.add_argument('--experiment-name', type=str, default='CBF Training',
                       help='MLflow experiment name')

    args = parser.parse_args()

    if args.train:
        # Setup MLflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment(args.experiment_name)
        
        with mlflow.start_run(run_name="CBF Training Run") as run:
            # Log parameters
            mlflow.log_params({
                "input_data": args.input,
                "serving_size": args.servings,
                "vectorizer_type": "TF-IDF",
                "max_features": 500
            })
            
            # Initialize and train model
            trainer = CBFTrainer(args.input)
            feature_matrix, simplified_df = trainer.train()
            
            # Log metrics
            num_recipes = len(simplified_df)
            num_features = feature_matrix.shape[1]
            mlflow.log_metrics({
                "num_recipes": num_recipes,
                "num_features": num_features,
                "avg_calories": simplified_df['nutrition'].apply(lambda x: x['calories']).mean(),
                "avg_protein": simplified_df['nutrition'].apply(lambda x: x['protein']).mean()
            })
            
            # Log artifacts
            mlflow.log_artifacts(trainer.model_dir, "model_artifacts")
            mlflow.log_text(str(simplified_df.head().to_dict()), "data_sample.json")
            
            print(f"Training completed! Run ID: {run.info.run_id}")

        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} not found")
            return

        print(f"Training CBF model using {args.input}...")
        features, df = train_cbf_model(args.input, servings=args.servings)
        print(f"Model saved to models/ directory")

    if args.recommend:
        if args.exclude:
            print(f"Excluding ingredients: {args.exclude}")
        if args.diet_type:
            print(f"Dietary preference: {args.diet_type}")

        print(f"Finding recipes similar to: {args.recommend}")
        recommendations = get_recommendations(
            args.recommend,
            n=args.count,
            max_calories=args.max_calories,
            excluded_ingredients=args.exclude,
            dietary_type=args.diet_type
        )

        if recommendations:
            print("\nRecommendations:")
            for rec in recommendations:
                print(f"ID: {rec['ID']}")
                print(f"Title: {rec['Title']}")
                print(f"Calories: {rec['Nutrition']['calories']}")
                print(f"Protein: {rec['Nutrition']['protein']}g")
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