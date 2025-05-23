
import argparse
import json
import sys
import os
import mlflow
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.pso_meal_scheduler import MealScheduler


def calculate_average_nutrition(meal_plan):
    """
    Calculate the correct average daily nutrition from a meal plan
    """
    daily_values = []

    # Extract daily nutrition values from each day
    for day in meal_plan['meal_plan']:
        daily_values.append(day['daily_nutrition'])

    # Calculate average across days
    average = {
        'calories': sum(day['calories'] for day in daily_values) / len(daily_values),
        'protein': sum(day['protein'] for day in daily_values) / len(daily_values),
        'fat': sum(day['fat'] for day in daily_values) / len(daily_values),
        'carbohydrates': sum(day['carbohydrates'] for day in daily_values) / len(daily_values),
        'fiber': sum(day['fiber'] for day in daily_values) / len(daily_values),
        # 'calcium': sum(day.get('calcium', 0) for day in daily_values) / len(daily_values)
    }

    return average


def main():
    parser = argparse.ArgumentParser(description='Generate optimized meal plans using PSO')

    # User profile arguments
    parser.add_argument('--age', type=int, required=True, help='User age in years')
    parser.add_argument('--gender', type=str, required=True, choices=['male', 'female'],
                        help='User gender')
    parser.add_argument('--weight', type=float, required=True, help='User weight in kg')
    parser.add_argument('--height', type=float, required=True, help='User height in cm')
    parser.add_argument('--activity-level', type=str, required=True,
                        choices=['sedentary', 'lightly_active', 'moderately_active',
                                 'very_active', 'extra_active'],
                        help='User activity level')

    # Meal plan settings
    parser.add_argument('--meals-per-day', type=int, default=3, choices=[1, 2, 3, 4],
                        help='Number of meals per day (1-4)')
    parser.add_argument('--recipes-per-meal', type=int, default=3, choices=[1, 2, 3, 4, 5],
                        help='Number of recipes to combine per meal (1-5)')
    parser.add_argument('--goal', type=str, default='maintain',
                        choices=['lose', 'maintain', 'gain'],
                        help='Weight goal')

    # User preferences
    parser.add_argument('--exclude', nargs='+', help='Ingredients to exclude (e.g., chicken beef)')
    parser.add_argument('--diet-type', choices=['vegetarian', 'vegan', 'pescatarian'],
                        help='Dietary preference')

    # Output options
    parser.add_argument('--output', type=str, default='output/meal_plan.json',
                        help='Output file path')

    args = parser.parse_args()

    args.output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Setup MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Meal Scheduler PSO")

    try:
        with mlflow.start_run(run_name="PSO Meal Plan"):
            # Log parameters
            mlflow.log_params({
                "age": args.age,
                "gender": args.gender,
                "weight": args.weight,
                "height": args.height,
                "activity_level": args.activity_level,
                "meals_per_day": args.meals_per_day,
                "recipes_per_meal": args.recipes_per_meal,
                "goal": args.goal,
                "diet_type": args.diet_type or "none",
                "excluded_ingredients": ",".join(args.exclude) if args.exclude else "none"
            })

            # Create meal scheduler
            scheduler = MealScheduler()

            # Create user params dictionary
            user_params = {
                'age': args.age,
                'gender': args.gender,
                'weight': args.weight,
                'height': args.height,
                'activity_level': args.activity_level,
                'goal': args.goal,
                'meals_per_day': args.meals_per_day
            }

            # Generate meal plan
            print("Generating optimized meal plan...")
            meal_plan = scheduler.generate_meal_plan(
                age=args.age,
                gender=args.gender,
                weight=args.weight,
                height=args.height,
                activity_level=args.activity_level,
                meals_per_day=args.meals_per_day,
                recipes_per_meal=args.recipes_per_meal,
                goal=args.goal,
                excluded_ingredients=args.exclude,
                dietary_type=args.diet_type
            )

            # Calculate average nutrition
            corrected_average = calculate_average_nutrition(meal_plan)
            meal_plan['average_daily_nutrition'] = corrected_average

            # Save meal plan
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            scheduler.save_meal_plan(meal_plan, args.output)

            # Log metrics and artifacts
            mlflow.log_metrics({
                "calories": corrected_average['calories'],
                "protein": corrected_average['protein'],
                "fat": corrected_average['fat'],
                "carbohydrates": corrected_average['carbohydrates'],
                "fiber": corrected_average['fiber']
            })

            mlflow.log_artifact(args.output)
            mlflow.set_tag("status", "success")

            # Print summary
            print("\nMeal Plan Summary:")
            print(f"Days: 7, Meals per day: {args.meals_per_day}, Recipes per meal: {args.recipes_per_meal}")
            print(f"Average daily nutrition:")
            nutrition = meal_plan['average_daily_nutrition']
            print(f"  - Calories: {nutrition['calories']:.0f} kcal")
            print(f"  - Protein: {nutrition['protein']:.1f} g")
            print(f"  - Fat: {nutrition['fat']:.1f} g")
            print(f"  - Carbohydrates: {nutrition['carbohydrates']:.1f} g")
            print(f"  - Fiber: {nutrition['fiber']:.1f} g")
            # print(f"  - Calcium: {nutrition.get('calcium', 0):.1f} mg")

            print(f"\nTarget nutrition:")
            targets = meal_plan['target_nutrition']
            print(f"  - Calories: {targets['calories']:.0f} kcal")
            print(f"  - Protein: {targets['protein']:.1f} g")
            print(f"  - Fat: {targets['fat']:.1f} g")
            print(f"  - Carbohydrates: {targets['carbohydrates']:.1f} g")
            print(f"  - Fiber: {targets['fiber']:.1f} g")
            # print(f"  - Calcium: {targets.get('calcium', 0):.1f} mg")

            print(f"\nComplete meal plan saved to {args.output}")

    except Exception as e:
        print(f"An error occurred: {e}")
        mlflow.set_tag("status", "failed")
        mlflow.log_param("error_message", str(e))
        raise

if __name__ == "__main__":
    main()