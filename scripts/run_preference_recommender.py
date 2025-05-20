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

    # Add user parameters for portion adjustment
    parser.add_argument('--age', type=int, help='User age in years')
    parser.add_argument('--gender', type=str, choices=['male', 'female'], help='User gender')
    parser.add_argument('--weight', type=float, help='User weight in kg')
    parser.add_argument('--height', type=float, help='User height in cm')
    parser.add_argument('--activity-level', type=str,
                        choices=['sedentary', 'lightly_active', 'moderately_active', 'very_active', 'extra_active'],
                        help='User activity level')
    parser.add_argument('--goal', type=str, default='maintain',
                        choices=['lose', 'maintain', 'gain'], help='Weight goal')
    parser.add_argument('--meals-per-day', type=int, default=3, choices=[1, 2, 3, 4],
                        help='Number of meals per day')

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

    # Create user params dict if user parameters are provided
    user_params = None
    if args.age and args.gender and args.weight and args.height and args.activity_level:
        user_params = {
            'age': args.age,
            'gender': args.gender,
            'weight': args.weight,
            'height': args.height,
            'activity_level': args.activity_level,
            'goal': args.goal,
            'meals_per_day': args.meals_per_day
        }
        print("User profile detected, portions will be adjusted accordingly.")

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
    if user_params:
        print(
            f"- User profile: {args.age} years, {args.gender}, {args.weight}kg, {args.height}cm, {args.activity_level}")

    if base_ids:
        print(f"\nFinding recipes similar to: {base_ids}")
        recommendations = recommender.recommend(
            base_ids,
            n=args.count,
            max_calories=args.max_calories,
            user_preferences=user_preferences,
            user_params=user_params  # Pass user params for portion adjustment
        )

        if not recommendations.empty:
            print("\nRecommendations based on your preferences:")
            for _, row in recommendations.iterrows():
                print(f"ID: {row['ID']}")
                print(f"Title: {row['Title']}")

                # Print adjusted nutrition if available
                if 'Adjusted_Total_Nutrition' in row:
                    print(f"Nutrition (adjusted for {args.gender}, {args.age} years): " +
                          f"Calories={row['Adjusted_Total_Nutrition']['calories']:.1f}, " +
                          f"Protein={row['Adjusted_Total_Nutrition']['protein']:.1f}g, " +
                          f"Carbs={row['Adjusted_Total_Nutrition']['carbohydrates']:.1f}g, " +
                          f"Fat={row['Adjusted_Total_Nutrition']['fat']:.1f}g")
                else:
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
                    'Nutrition': row.get('Adjusted_Total_Nutrition', row['nutrition']),
                    'Similarity': float(row['similarity'])
                }
                # Add portion adjustment info if available
                if 'Portion_Adjustment' in row:
                    rec['Portion_Info'] = row['Portion_Adjustment']

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