import mlflow
import pandas as pd
import json
import os
from flask import Flask, request, jsonify
from src.models.pso_meal_scheduler import MealScheduler
from src.models.cbf_recommender import CBFRecommender
import argparse

def train_cbf_model(data_path, output_dir='models/'):
    """Train the content-based filtering model"""
    from src.models.cbf_trainer import CBFTrainer
    
    print(f"Training CBF model using data from {data_path}")
    trainer = CBFTrainer(data_path)
    feature_matrix, simplified_df = trainer.train()
    
    print(f"CBF model training completed. Feature matrix shape: {feature_matrix.shape}")
    print(f"Model artifacts saved to {output_dir}")
    
    return feature_matrix, simplified_df

def init_mlflow(tracking_uri):
    """Initialize MLflow tracking"""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Meal Planning")
    print(f"MLflow tracking initialized. URI: {tracking_uri}")

def test_meal_plan_generation(model_dir='models/'):
    """Test meal plan generation with sample user data"""
    scheduler = MealScheduler(model_dir=model_dir)
    
    # Sample user data
    user_data = {
        'age': 16,
        'gender': 'male',
        'weight': 65,  # kg
        'height': 175,  # cm
        'activity_level': 'moderately_active',
        'goal': 'maintain',
        'meals_per_day': 3,
        'recipes_per_meal': 1,
        'excluded_ingredients': ['pork', 'beef'],
        'dietary_type': None
    }
    
    print("Generating test meal plan...")
    
    # Create user preferences
    user_preferences = scheduler.create_user_preferences(
        excluded_ingredients=user_data['excluded_ingredients'],
        dietary_type=user_data['dietary_type']
    )
    
    # Generate meal plan
    meal_plan = scheduler.generate_meal_plan(
        age=user_data['age'],
        gender=user_data['gender'],
        weight=user_data['weight'],
        height=user_data['height'],
        activity_level=user_data['activity_level'],
        meals_per_day=user_data['meals_per_day'],
        recipes_per_meal=user_data['recipes_per_meal'],
        goal=user_data['goal'],
        excluded_ingredients=user_data['excluded_ingredients'],
        dietary_type=user_data['dietary_type']
    )
    
    # Save meal plan
    output_file = os.path.join('output', 'test_meal_plan.json')
    scheduler.save_meal_plan(meal_plan, output_file)
    
    print(f"Test meal plan generated and saved to {output_file}")
    
    return meal_plan

def test_recommendation_engine(model_dir='models/'):
    """Test the recommendation engine with sample recipes"""
    recommender = CBFRecommender(model_dir=model_dir)
    
    # Get sample recipe IDs
    sample_ids = recommender.meal_data['ID'].sample(2).tolist()
    
    print(f"Testing recommendations based on recipes: {sample_ids}")
    
    # Get recommendations
    recommendations = recommender.recommend(
        recipe_ids=sample_ids,
        n=5
    )
    
    if not recommendations.empty:
        print(f"Found {len(recommendations)} recommendations")
        print("Sample recommendations:")
        for _, row in recommendations.head(3).iterrows():
            print(f"- {row['Title']} (ID: {row['ID']}, Similarity: {row.get('similarity', 'N/A')})")
    else:
        print("No recommendations found")
    
    return recommendations

def initialize_api(host='0.0.0.0', port=5000):
    """Initialize and run the Flask API"""
    from app import app
    print(f"Starting API server on {host}:{port}")
    app.run(host=host, port=port, debug=True)

def prepare_data(input_file, output_file):
    """Process and enrich food data"""
    from src.data.nutrition_enricher import process_dataset
    
    print(f"Processing data from {input_file}")
    
    # Load recipe data
    recipe_df = pd.read_json(input_file)
    
    # Load nutrition data (should be provided)
    nutrition_file = os.path.join('data', 'nutrition_database.json')
    if os.path.exists(nutrition_file):
        nutrition_df = pd.read_json(nutrition_file)
    else:
        raise FileNotFoundError(f"Nutrition database not found: {nutrition_file}")
    
    # Process dataset
    enriched_df = process_dataset(recipe_df, nutrition_df)
    
    # Save processed data
    enriched_df.to_json(output_file, orient='records', indent=2)
    
    print(f"Data processing completed. Saved to {output_file}")
    print(f"Processed {len(enriched_df)} recipes")
    
    return enriched_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Food Recommendation System CLI')
    parser.add_argument('command', choices=['train', 'test-plan', 'test-recommendations', 'process-data', 'run-api'],
                      help='Command to execute')
    parser.add_argument('--input', help='Input file path')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--host', default='0.0.0.0', help='API host')
    parser.add_argument('--port', type=int, default=5000, help='API port')
    
    args = parser.parse_args()
    
    # Initialize MLflow
    init_mlflow("http://localhost:5000")
    
    if args.command == 'train':
        # Train CBF model
        if not args.input:
            args.input = 'data/processed/enriched_recipes.json'
        train_cbf_model(args.input)
    
    elif args.command == 'test-plan':
        # Test meal plan generation
        test_meal_plan_generation()
    
    elif args.command == 'test-recommendations':
        # Test recommendation engine
        test_recommendation_engine()
    
    elif args.command == 'process-data':
        # Process and enrich data
        if not args.input or not args.output:
            print("Error: Both --input and --output are required for data processing")
        else:
            prepare_data(args.input, args.output)
    
    elif args.command == 'run-api':
        # Run API server
        initialize_api(host=args.host, port=args.port)