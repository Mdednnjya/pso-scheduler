import os
import argparse
import json
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import CBF modules
from src.models.cbf_recommender import CBFRecommender
from src.models.user_preferences import UserPreferences


def main():
    parser = argparse.ArgumentParser(description='Preference-Based Recipe Recommender')
    parser.add_argument('--base-id', nargs='+', type=str, help='Base recipe IDs for recommendation')
    parser.add_argument('--exclude', nargs='+', help='Ingredients to exclude (e.g., chicken beef)')
    parser.add_argument('--diet-type', choices=['vegetarian', 'vegan', 'pescatarian'],
                        help='Dietary preference')
    parser.add_argument('--min-protein', type=float, help='Minimum protein content (g)')
    parser.add_argument('--max-calories', type=float, help='Maximum calories')
    parser.add_argument('--count', type=int, default=5, help='Number of recommendations to return')

    args = parser.parse_args()

    # Check if models exist
    model_dir = 'models/'
    if not os.path.exists(os.path.join(model_dir, 'meal_data.json')):
        print("Error: Model not found. Please train the model first.")
        return

    # Create user preferences
    min_nutrition = {}
    max_nutrition = {}

    if args.min_protein:
        min_nutrition['protein'] = args.min_protein

    if args.max_calories:
        max_nutrition['calories'] = args.max_calories

    user_preferences = UserPreferences(
        excluded_ingredients=args.exclude,
        dietary_type=args.diet_type,
        min_nutrition=min_nutrition,
        max_nutrition=max_nutrition
    )

    # Initialize recommender
    recommender = CBFRecommender(model_dir)

    # Get recommendations
    base_ids = args.base_id if args.base_id else []

    print("User Preferences:")
    if args.exclude:
        print(f"- Excluded ingredients: {args.exclude}")
    if args.diet_type:
        print(f"- Dietary type: {args.diet_type}")
    if args.min_protein:
        print(f"- Minimum protein: {args.min_protein}g")
    if args.max_calories:
        print(f"- Maximum calories: {args.max_calories}")

    if base_ids:
        print(f"\nFinding recipes similar to: {base_ids}")
        recommendations = recommender.recommend(
            base_ids,
            n=args.count,
            max_calories=args.max_calories,
            user_preferences=user_preferences
        )

        if not recommendations.empty:
            print("\nRecommendations based on your preferences:")
            for _, row in recommendations.iterrows():
                print(f"ID: {row['ID']}")
                print(f"Title: {row['Title']}")
                print(f"Nutrition: Calories={row['nutrition']['calories']:.1f}, " +
                      f"Protein={row['nutrition']['protein']:.1f}g, " +
                      f"Carbs={row['nutrition']['carbohydrates']:.1f}g, " +
                      f"Fat={row['nutrition']['fat']:.1f}g")
                print(f"Similarity: {row['similarity']:.4f}")
                print("-" * 40)

            # Save recommendations to file
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)

            # Prepare for JSON serialization
            recommendations_list = []
            for _, row in recommendations.iterrows():
                rec = {
                    'ID': row['ID'],
                    'Title': row['Title'],
                    'Nutrition': row['nutrition'],
                    'Similarity': float(row['similarity'])
                }
                recommendations_list.append(rec)

            with open(os.path.join(output_dir, "preference_recommendations.json"), 'w') as f:
                json.dump(recommendations_list, f, indent=2)

            print(f"Recommendations saved to output/preference_recommendations.json")
        else:
            print("No recommendations found that match your preferences.")
    else:
        print("Please provide base recipe IDs using --base-id parameter.")

        # Show some available recipes that match the preferences
        with open(os.path.join(model_dir, 'meal_data.json'), 'r', encoding='utf-8') as f:
            meal_data = pd.DataFrame(json.load(f))

        filtered_data = user_preferences.filter_recipes(meal_data)

        if not filtered_data.empty:
            print("\nSome recipes that match your preferences:")
            sample_size = min(5, len(filtered_data))
            for _, row in filtered_data.sample(sample_size).iterrows():
                print(f"ID: {row['ID']} - {row['Title']}")

            print(f"\nTotal matching recipes: {len(filtered_data)}")
            print("Use --base-id with one of these IDs to get similar recommendations.")
        else:
            print("No recipes match your preferences. Try relaxing some constraints.")


if __name__ == "__main__":
    main()