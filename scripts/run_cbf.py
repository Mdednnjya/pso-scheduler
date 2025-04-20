import argparse
import os
import json
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.models.cbf_trainer import CBFTrainer
from src.models.cbf_recommender import CBFRecommender
from src.data.data_loader import load_data
from src.data.nutrition_enricher import process_dataset
from src.data.export_utils import export_to_json, export_summary_csv

def main():
    parser = argparse.ArgumentParser(description='Recipe Recommendation System')
    parser.add_argument('--train', action='store_true', help='Train the CBF model')
    parser.add_argument('--recommend', nargs='+', type=str, help='Get recommendations based on recipe IDs')
    parser.add_argument('--max-calories', type=float, help='Maximum calories for recommendations')
    parser.add_argument('--count', type=int, default=5, help='Number of recommendations to return')
    
    args = parser.parse_args()
    
    # Define paths
    data_dir = "data"
    output_dir = "output"
    enriched_json_path = os.path.join(output_dir, "enriched_recipes.json")
    
    if args.train:
        print("Processing nutritional data...")
        # Check if enriched data exists, if not create it
        if not os.path.exists(enriched_json_path):
            # Load and process data
            recipe_path = os.path.join(data_dir, "combined_dataset.csv")
            nutrition_path = os.path.join(data_dir, "tkpi_data.csv")
            
            recipes_df, nutrition_df = load_data(recipe_path, nutrition_path)
            
            # Ensure nutrition values are numeric
            numeric_columns = ['calories', 'protein', 'fat', 'carbohydrates', 'fiber', 'calcium']
            for col in numeric_columns:
                nutrition_df[col] = pd.to_numeric(nutrition_df[col], errors='coerce').fillna(0)
            
            # Process dataset
            enriched_df = process_dataset(recipes_df, nutrition_df)
            
            # Export results
            os.makedirs(output_dir, exist_ok=True)
            export_to_json(enriched_df, enriched_json_path)
            export_summary_csv(enriched_df, os.path.join(output_dir, "enriched_recipes.csv"))
        
        print("Training recommendation model...")
        trainer = CBFTrainer(enriched_json_path)
        feature_matrix, simplified_df = trainer.train()
        
        # Show feature shape
        print(f"Feature matrix shape: {feature_matrix.shape}")
        print(f"Number of recipes: {len(simplified_df)}")
    
    if args.recommend:
        if not os.path.exists(os.path.join("models", "feature_matrix.pkl")):
            print("Error: Model not trained. Please run with --train first.")
            return
        
        print("Loading recommendation model...")
        recommender = CBFRecommender()
        
        print(f"Finding recipes similar to IDs: {args.recommend}")
        recommendations = recommender.recommend(
            args.recommend, 
            n=args.count,
            max_calories=args.max_calories
        )
        
        if recommendations.empty:
            print("No recommendations found. The provided recipe IDs may not exist in the dataset.")
        else:
            print("\nRecommendations:")
            for _, row in recommendations.iterrows():
                nutrients = row['nutrition']
                print(f"ID: {row['ID']}")
                print(f"Title: {row['Title']}")
                print(f"Calories: {nutrients['calories']:.1f} kcal")
                print(f"Protein: {nutrients['protein']:.1f} g")
                print(f"Carbs: {nutrients['carbohydrates']:.1f} g")
                print(f"Fat: {nutrients['fat']:.1f} g")
                print("-" * 40)
            
            # Save recommendations to file
            recommendations_output = os.path.join(output_dir, "recommendations.json")
            recommendations.to_json(recommendations_output, orient='records', indent=2)
            print(f"Recommendations saved to {recommendations_output}")

if __name__ == "__main__":
    main()