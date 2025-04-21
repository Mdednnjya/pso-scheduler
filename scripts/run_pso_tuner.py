import os
import argparse
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.pso_meal_scheduler import MealScheduler
from scripts.run_meal_evaluator import MealPlanEvaluator


class PSOParameterTuner:
    def __init__(self, output_dir='output/tuning'):
        """Initialize the PSO parameter tuner"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []

    def run_experiment(self, params, user_profile, preferences):
        """Run a PSO experiment with specific parameters"""
        scheduler = MealScheduler()

        # Generate meal plan with given parameters
        meal_plan = scheduler.generate_meal_plan(
            age=user_profile['age'],
            gender=user_profile['gender'],
            weight=user_profile['weight'],
            height=user_profile['height'],
            activity_level=user_profile['activity_level'],
            meals_per_day=params['meals_per_day'],
            recipes_per_meal=params['recipes_per_meal'],
            goal=user_profile.get('goal', 'maintain'),
            excluded_ingredients=preferences.get('excluded_ingredients'),
            dietary_type=preferences.get('dietary_type')
        )

        # Save the meal plan
        output_file = os.path.join(
            self.output_dir,
            f"meal_plan_m{params['meals_per_day']}_r{params['recipes_per_meal']}.json"
        )
        scheduler.save_meal_plan(meal_plan, output_file)

        # Evaluate the meal plan
        evaluator = MealPlanEvaluator(output_file)
        eval_results = evaluator.evaluate()

        # Store results
        experiment = {
            'parameters': params,
            'evaluation': eval_results,
            'file': output_file
        }
        self.results.append(experiment)

        return experiment

    def grid_search(self, param_grid, user_profile, preferences):
        """Perform grid search over parameter combinations"""
        # Generate all parameter combinations
        param_names = param_grid.keys()
        param_values = param_grid.values()
        param_combinations = [dict(zip(param_names, combo))
                              for combo in product(*param_values)]

        print(f"Running grid search with {len(param_combinations)} parameter combinations")

        # Run experiments for each combination
        for i, params in enumerate(param_combinations):
            print(f"\nExperiment {i + 1}/{len(param_combinations)}:")
            print(f"Parameters: {params}")

            experiment = self.run_experiment(params, user_profile, preferences)
            print(f"Overall Score: {experiment['evaluation']['summary']['overall_score']:.2f}%")

        # Find best parameters
        best_experiment = max(self.results,
                              key=lambda x: x['evaluation']['summary']['overall_score'])

        print("\nBest Parameter Combination:")
        print(f"Parameters: {best_experiment['parameters']}")
        print(f"Overall Score: {best_experiment['evaluation']['summary']['overall_score']:.2f}%")

        return best_experiment

    def generate_comparison_plots(self):
        """Generate plots comparing different parameter combinations"""
        if not self.results:
            print("No results to plot")
            return []

        # Extract data for plotting
        data = []
        for result in self.results:
            params = result['parameters']
            eval_data = result['evaluation']['summary']

            row = {
                'meals_per_day': params['meals_per_day'],
                'recipes_per_meal': params['recipes_per_meal'],
                'overall_score': eval_data['overall_score'],
                'nutritional_accuracy': eval_data['overall_nutritional_accuracy'],
                'variety_score': eval_data['variety_score'],
                'consistency_score': eval_data['consistency_score']
            }
            data.append(row)

        df = pd.DataFrame(data)

        # Create parameter combination labels
        df['params'] = df.apply(
            lambda x: f"M{x['meals_per_day']}/R{x['recipes_per_meal']}",
            axis=1
        )

        # 1. Overall scores comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        df.plot(x='params', y='overall_score', kind='bar', ax=ax, color='skyblue')
        ax.set_title('Overall Meal Plan Quality by Parameter Combination')
        ax.set_xlabel('Parameter Combination (Meals/Recipes)')
        ax.set_ylabel('Overall Score (%)')
        plt.tight_layout()
        overall_plot = os.path.join(self.output_dir, 'overall_scores.png')
        plt.savefig(overall_plot)

        # 2. Component scores comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        df.plot(
            x='params',
            y=['nutritional_accuracy', 'variety_score', 'consistency_score'],
            kind='bar',
            ax=ax
        )
        ax.set_title('Score Components by Parameter Combination')
        ax.set_xlabel('Parameter Combination (Meals/Recipes)')
        ax.set_ylabel('Score (%)')
        plt.tight_layout()
        components_plot = os.path.join(self.output_dir, 'score_components.png')
        plt.savefig(components_plot)

        # Save comparison data
        comparison_file = os.path.join(self.output_dir, 'parameter_comparison.csv')
        df.to_csv(comparison_file, index=False)

        return [overall_plot, components_plot, comparison_file]


def main():
    parser = argparse.ArgumentParser(description='Tune PSO meal scheduler parameters')

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
    parser.add_argument('--goal', type=str, default='maintain',
                        choices=['lose', 'maintain', 'gain'],
                        help='Weight goal')

    # User preferences
    parser.add_argument('--exclude', nargs='+', help='Ingredients to exclude (e.g., chicken beef)')
    parser.add_argument('--diet-type', choices=['vegetarian', 'vegan', 'pescatarian'],
                        help='Dietary preference')

    # Output options
    parser.add_argument('--output-dir', type=str, default='output/tuning',
                        help='Output directory for tuning results')

    args = parser.parse_args()

    # Set up user profile and preferences
    user_profile = {
        'age': args.age,
        'gender': args.gender,
        'weight': args.weight,
        'height': args.height,
        'activity_level': args.activity_level,
        'goal': args.goal
    }

    preferences = {
        'excluded_ingredients': args.exclude,
        'dietary_type': args.diet_type
    }

    # Parameter grid to search
    param_grid = {
        'meals_per_day': [2, 3, 4],
        'recipes_per_meal': [1, 2, 3]
    }

    # Run parameter tuning
    tuner = PSOParameterTuner(output_dir=args.output_dir)
    best_params = tuner.grid_search(param_grid, user_profile, preferences)

    # Generate comparison plots
    plot_files = tuner.generate_comparison_plots()
    print(f"\nGenerated comparison plots: {', '.join(plot_files)}")

    print(f"\nTuning complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()