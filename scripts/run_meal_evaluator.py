import os
import argparse
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.pso_meal_scheduler import MealScheduler


class MealPlanEvaluator:
    def __init__(self, meal_plan_file):
        """Initialize evaluator with a meal plan JSON file"""
        with open(meal_plan_file, 'r') as f:
            self.meal_plan = json.load(f)

    def calculate_mape(self):
        """Calculate Mean Absolute Percentage Error for nutritional targets"""
        target = self.meal_plan['target_nutrition']
        actual = self.meal_plan['average_daily_nutrition']

        mape_values = {}
        for nutrient in target:
            if target[nutrient] > 0:  # Avoid division by zero
                mape_values[nutrient] = abs(actual[nutrient] - target[nutrient]) / target[nutrient] * 100

        # Overall MAPE (average across nutrients)
        mape_values['overall'] = np.mean(list(mape_values.values()))
        return mape_values

    def calculate_achievement_ratios(self):
        """Calculate achievement ratio (actual/target) for each nutrient"""
        target = self.meal_plan['target_nutrition']
        actual = self.meal_plan['average_daily_nutrition']

        ratios = {}
        for nutrient in target:
            if target[nutrient] > 0:  # Avoid division by zero
                ratios[nutrient] = actual[nutrient] / target[nutrient]

        return ratios

    def calculate_recipe_variety(self):
        """Calculate metrics for meal variety"""
        # Extract all recipe IDs
        all_recipes = []
        unique_recipes = set()

        for day in self.meal_plan['meal_plan']:
            for meal in day['meals']:
                for recipe in meal['recipes']:
                    all_recipes.append(recipe['meal_id'])
                    unique_recipes.add(recipe['meal_id'])

        # Unique recipe ratio
        variety_ratio = len(unique_recipes) / len(all_recipes) if all_recipes else 0

        # Shannon entropy (diversity measure)
        recipe_counts = {}
        for recipe_id in all_recipes:
            recipe_counts[recipe_id] = recipe_counts.get(recipe_id, 0) + 1

        probabilities = [count / len(all_recipes) for count in recipe_counts.values()]
        shannon_entropy = -sum(p * np.log2(p) for p in probabilities)

        # Maximum possible entropy (if all recipes were used equally)
        max_entropy = np.log2(len(all_recipes)) if all_recipes else 0
        normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0

        return {
            'unique_ratio': variety_ratio,
            'shannon_entropy': shannon_entropy,
            'normalized_entropy': normalized_entropy,
            'recipe_repetition': {id: count for id, count in recipe_counts.items() if count > 1}
        }

    def calculate_daily_consistency(self):
        """Evaluate consistency of nutrition across days"""
        daily_nutrition = []

        for day in self.meal_plan['meal_plan']:
            daily_nutrition.append(day['daily_nutrition'])

        # Calculate coefficient of variation for each nutrient
        cv_values = {}
        for nutrient in daily_nutrition[0]:
            values = [day[nutrient] for day in daily_nutrition]
            mean = np.mean(values)
            std = np.std(values)
            cv_values[nutrient] = (std / mean) * 100 if mean > 0 else 0

        return cv_values

    def generate_plots(self, output_dir='data/plots/pso'):
        """Generate evaluation plots"""
        os.makedirs(output_dir, exist_ok=True)

        # 1. Target vs. Actual Nutrition
        target = self.meal_plan['target_nutrition']
        actual = self.meal_plan['average_daily_nutrition']

        nutrients = list(target.keys())
        x = np.arange(len(nutrients))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width / 2, [target[n] for n in nutrients], width, label='Target')
        ax.bar(x + width / 2, [actual[n] for n in nutrients], width, label='Actual')

        ax.set_title('Target vs. Actual Daily Nutrition')
        ax.set_xticks(x)
        ax.set_xticklabels(nutrients)
        ax.legend()

        plt.savefig(f"{output_dir}/nutrition_comparison.png")

        # 2. Daily Nutrition Variation
        daily_data = []
        days = []

        for i, day in enumerate(self.meal_plan['meal_plan']):
            daily_data.append(day['daily_nutrition'])
            days.append(f"Day {i + 1}")

        df = pd.DataFrame(daily_data, index=days)

        fig, ax = plt.subplots(figsize=(14, 8))
        df.plot(kind='bar', ax=ax)
        ax.set_title('Daily Nutritional Variation')
        ax.set_ylabel('Amount')
        plt.tight_layout()

        plt.savefig(f"{output_dir}/daily_variation.png")

        return [f"{output_dir}/nutrition_comparison.png", f"{output_dir}/daily_variation.png"]

    def evaluate(self):
        """Run all evaluations and return results"""
        results = {
            'nutritional_accuracy': {
                'mape': self.calculate_mape(),
                'achievement_ratios': self.calculate_achievement_ratios()
            },
            'variety_metrics': self.calculate_recipe_variety(),
            'consistency_metrics': self.calculate_daily_consistency(),
        }

        # Add summary scores
        results['summary'] = {
            'overall_nutritional_accuracy': 100 - results['nutritional_accuracy']['mape']['overall'],
            'variety_score': results['variety_metrics']['normalized_entropy'] * 100,
            'consistency_score': 100 - np.mean(list(results['consistency_metrics'].values()))
        }

        # Weighted overall score
        results['summary']['overall_score'] = (
                0.6 * results['summary']['overall_nutritional_accuracy'] +
                0.3 * results['summary']['variety_score'] +
                0.1 * results['summary']['consistency_score']
        )

        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate PSO meal scheduler results')
    parser.add_argument('--meal-plan', type=str, default='output/meal_plan.json',
                        help='Path to meal plan JSON file')
    parser.add_argument('--output', type=str, default='output/evaluation_results.json',
                        help='Path to output evaluation results')
    parser.add_argument('--generate-plots', action='store_true',
                        help='Generate evaluation plots')

    args = parser.parse_args()

    # Check if meal plan file exists
    if not os.path.exists(args.meal_plan):
        print(f"Error: Meal plan file {args.meal_plan} not found.")
        return

    # Evaluate meal plan
    evaluator = MealPlanEvaluator(args.meal_plan)
    results = evaluator.evaluate()

    # Print summary
    print("\nMeal Plan Evaluation Summary:")
    print("-" * 40)
    print(f"Overall Score: {results['summary']['overall_score']:.2f}%")
    print(f"Nutritional Accuracy: {results['summary']['overall_nutritional_accuracy']:.2f}%")
    print(f"Variety Score: {results['summary']['variety_score']:.2f}%")
    print(f"Consistency Score: {results['summary']['consistency_score']:.2f}%")
    print("-" * 40)

    # Print nutritional details
    print("\nNutritional Achievement:")
    for nutrient, ratio in results['nutritional_accuracy']['achievement_ratios'].items():
        target = evaluator.meal_plan['target_nutrition'][nutrient]
        actual = evaluator.meal_plan['average_daily_nutrition'][nutrient]
        print(f"  {nutrient.capitalize()}: {actual:.1f}/{target:.1f} ({ratio * 100:.1f}%)")

    # Generate plots if requested
    if args.generate_plots:
        plot_files = evaluator.generate_plots()
        print(f"\nGenerated evaluation plots: {', '.join(plot_files)}")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed evaluation results saved to {args.output}")


if __name__ == "__main__":
    main()