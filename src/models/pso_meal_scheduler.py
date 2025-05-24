# src/models/pso_meal_scheduler.py
import numpy as np
import random
import json
import os
import mlflow

from src.models.portion_adjuster import get_user_nutrition_targets
from src.models.user_preferences import UserPreferences
from src.models.cbf_recommender import CBFRecommender


class ParticleSwarmOptimizer:
    def __init__(self,
                 num_particles=30,
                 num_days=7,
                 meals_per_day=3,
                 recipes_per_meal=3,
                 max_iterations=100,
                 w=0.7,
                 c1=1.5,
                 c2=1.5):
        """
        Initialize the PSO algorithm for meal scheduling

        Args:
            num_particles: Number of particles in the swarm
            num_days: Number of days to schedule (default: 7)
            meals_per_day: Number of meals per day (default: 3)
            recipes_per_meal: Number of recipes to combine per meal (default: 3)
            max_iterations: Maximum number of iterations
            w: Inertia weight
            c1: Cognitive coefficient (personal best)
            c2: Social coefficient (global best)
        """
        self.num_particles = num_particles
        self.num_days = num_days
        self.meals_per_day = meals_per_day
        self.recipes_per_meal = recipes_per_meal  # New attribute
        self.max_iterations = max_iterations
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient
        self.mlflow_experiment = "Meal Planning Optimization"

        # Load meal data
        self.cbf_recommender = CBFRecommender()
        self.meal_data = self.cbf_recommender.meal_data
        self.meal_ids = self.meal_data['ID'].unique().tolist()

        # Parameters for the fitness function
        self.target_metrics = {
            'calories': 0,
            'protein': 0,
            'fat': 0,
            'carbohydrates': 0,
            'fiber': 0,
            # 'calcium': 0
        }

        # Weights for different nutrients in fitness calculation
        self.nutrient_weights = {
            'calories': 15.0,
            'protein': 2.0,
            'fat': 8.0,
            'carbohydrates': 3.0,
            'fiber': 1.5
        }

    def set_user_requirements(self, age, gender, weight, height, activity_level,
                              goal='maintain', user_preferences=None):
        """
        Set user requirements for meal optimization

        Args:
            age: User's age in years
            gender: User's gender ('male' or 'female')
            weight: User's weight in kg
            height: User's height in cm
            activity_level: Level of physical activity ('sedentary', 'lightly_active',
                            'moderately_active', 'very_active', 'extra_active')
            goal: Weight goal ('lose', 'maintain', 'gain')
            user_preferences: UserPreferences object for filtering recipes
        """
        with mlflow.start_run(nested=True, run_name="Nutrition Targets"):
            # Log user parameters
            mlflow.log_params({
                "age": age,
                "gender": gender,
                "weight": weight,
                "height": height,
                "activity_level": activity_level,
                "goal": goal
            })

            # Set the user preferences for filtering recipes
            self.user_preferences = user_preferences

            # Filter meals based on user preferences if provided
            if self.user_preferences:
                self.filtered_meal_data = self.user_preferences.filter_recipes(self.meal_data)
                if self.filtered_meal_data.empty:
                    raise ValueError("No recipes match the user preferences.")
                self.meal_ids = self.filtered_meal_data['ID'].unique().tolist()
            else:
                self.filtered_meal_data = self.meal_data

            # NEW: Apply calorie-based filtering
            self.filter_recipes_by_calorie_target()

            # Use portion_adjuster to calculate target metrics based on user parameters
            user_params = {
                'age': age,
                'gender': gender,
                'weight': weight,
                'height': height,
                'activity_level': activity_level,
                'goal': goal,
                'meals_per_day': self.meals_per_day
            }
            self.target_metrics = get_user_nutrition_targets(user_params)

            # Adaptive weights based on target calories (PSO best practice)
            target_calories = self.target_metrics['calories']

            if target_calories < 2000:  # Low calorie targets need more aggressive control
                self.nutrient_weights = {
                    'calories': 25.0,  # Very aggressive for low targets
                    'protein': 3.0,
                    'fat': 15.0,
                    'carbohydrates': 5.0,
                    'fiber': 2.0
                }
            elif target_calories > 3000:  # High calorie targets more flexible
                self.nutrient_weights = {
                    'calories': 8.0,  # Less aggressive for high targets
                    'protein': 2.0,
                    'fat': 6.0,
                    'carbohydrates': 2.0,
                    'fiber': 1.5
                }
            else:  # Medium targets (your current good config)
                self.nutrient_weights = {
                    'calories': 15.0,
                    'protein': 2.0,
                    'fat': 8.0,
                    'carbohydrates': 3.0,
                    'fiber': 1.5
                }

            # Log the target metrics
            mlflow.log_params(self.target_metrics)

            print(f"Daily targets set: Calories: {self.target_metrics['calories']:.0f}, "
                  f"Protein: {self.target_metrics['protein']:.1f}g, "
                  f"Fat: {self.target_metrics['fat']:.1f}g, "
                  f"Carbs: {self.target_metrics['carbohydrates']:.1f}g, "
                  f"Fiber: {self.target_metrics['fiber']:.1f}g")

    def calculate_nutritional_value(self, meal_schedule):
        """
        Calculate total nutritional value for a meal schedule
        """
        total_nutrition = {
            'calories': 0,
            'protein': 0,
            'fat': 0,
            'carbohydrates': 0,
            'fiber': 0,
            # 'calcium': 0
        }

        for day in meal_schedule:
            for meal in day:
                for meal_id in meal:
                    meal_record = self.filtered_meal_data[self.filtered_meal_data['ID'] == str(meal_id)]
                    if not meal_record.empty:
                        record = meal_record.iloc[0]

                        # Get basic nutrition
                        if 'Adjusted_Total_Nutrition' in record:
                            nutrition = record['Adjusted_Total_Nutrition']
                        else:
                            nutrition = record['nutrition']

                        total_nutrition['calories'] += nutrition.get('calories', 0)
                        total_nutrition['protein'] += nutrition.get('protein', 0)
                        total_nutrition['fat'] += nutrition.get('fat', 0)
                        total_nutrition['carbohydrates'] += nutrition.get('carbohydrates', 0)
                        total_nutrition['fiber'] += nutrition.get('fiber', 0)

                        # ADD: Calculate calcium from Ingredients_Enriched
                        if 'Ingredients_Enriched' in record:
                            ingredients = record['Ingredients_Enriched']
                            recipe_calcium = sum(float(ing.get('calcium', 0)) for ing in ingredients)
                            # total_nutrition['calcium'] += recipe_calcium

            # Return average per day
        days = len(meal_schedule)
        return {k: v / days for k, v in total_nutrition.items()}

    def calculate_meal_variety(self, meal_schedule):
        """
        Calculate the variety score for a meal schedule

        Args:
            meal_schedule: A 3D array representing meal IDs for each day, meal, and recipe

        Returns:
            A variety score (higher is better)
        """
        # Flatten the schedule to count unique meals
        flat_schedule = [recipe for day in meal_schedule for meal in day for recipe in meal]
        unique_meals = len(set(flat_schedule))

        # Calculate variety percentage (unique meals / total meals)
        total_meals = len(flat_schedule)
        variety_score = unique_meals / total_meals if total_meals > 0 else 0

        return variety_score

    def fitness_function(self, meal_schedule):
        """
        Calculate fitness score with exponential penalty (PSO best practice)
        Lower score is better
        """
        # Calculate average daily nutrition for the schedule
        daily_nutrition = self.calculate_nutritional_value(meal_schedule)

        # Calculate penalty for deviation from target with exponential scaling
        penalty = 0
        for nutrient, target in self.target_metrics.items():
            if target > 0 and nutrient in daily_nutrition:
                # Calculate relative error as percentage
                relative_error = abs(daily_nutrition[nutrient] - target) / target

                # Exponential penalty for large errors (PSO best practice)
                if relative_error > 0.5:  # > 50% error gets exponential penalty
                    exponential_penalty = relative_error ** 2.5  # Aggressive for large errors
                elif relative_error > 0.2:  # 20-50% error gets quadratic penalty
                    exponential_penalty = relative_error ** 2
                else:  # < 20% error gets linear penalty
                    exponential_penalty = relative_error

                # Apply weight to the penalty
                weighted_error = exponential_penalty * self.nutrient_weights.get(nutrient, 1.0)
                penalty += weighted_error

        # Calculate meal variety (higher is better)
        variety_score = self.calculate_meal_variety(meal_schedule)
        # Convert to penalty (lower is better)
        variety_penalty = 1 - variety_score

        # Combine nutrition and variety penalties
        # 85% weight to nutrition (higher than before), 15% to variety
        total_penalty = (0.85 * penalty) + (0.15 * variety_penalty)

        return total_penalty

    def initialize_swarm(self):
        """
        Initialize particles with random meal schedules

        Returns:
            Tuple of (positions, velocities, pbests, gbest)
        """
        # Initialize positions (meal schedules)
        positions = []
        for _ in range(self.num_particles):
            # Random meal schedule for each particle
            schedule = []
            for _ in range(self.num_days):
                day_meals = []
                for _ in range(self.meals_per_day):
                    # Random recipes for each meal
                    meal_recipes = random.choices(self.meal_ids, k=self.recipes_per_meal)
                    day_meals.append(meal_recipes)
                schedule.append(day_meals)
            positions.append(schedule)

        # Initialize velocities
        velocities = []
        for _ in range(self.num_particles):
            # Velocity is represented as probability of changing a recipe
            v = []
            for _ in range(self.num_days):
                day_v = []
                for _ in range(self.meals_per_day):
                    meal_v = [random.uniform(0, 0.3) for _ in range(self.recipes_per_meal)]
                    day_v.append(meal_v)
                v.append(day_v)
            velocities.append(v)

        # Initialize personal best positions and scores
        pbests = positions.copy()
        pbest_scores = [self.fitness_function(p) for p in pbests]

        # Initialize global best
        gbest_idx = np.argmin(pbest_scores)
        gbest = pbests[gbest_idx]
        gbest_score = pbest_scores[gbest_idx]

        return positions, velocities, pbests, pbest_scores, gbest, gbest_score

    def update_velocity(self, positions, velocities, pbests, gbest, particle_idx):
        """
        Update velocity for a single particle

        Args:
            positions: List of current positions
            velocities: List of current velocities
            pbests: List of personal best positions
            gbest: Global best position
            particle_idx: Index of the particle to update

        Returns:
            Updated velocity for the particle
        """
        new_velocity = []
        position = positions[particle_idx]
        velocity = velocities[particle_idx]
        pbest = pbests[particle_idx]

        for day_idx in range(len(position)):
            day_v = []
            for meal_idx in range(len(position[day_idx])):
                meal_v = []
                for recipe_idx in range(len(position[day_idx][meal_idx])):
                    # Get current components
                    current_recipe = position[day_idx][meal_idx][recipe_idx]
                    current_v = velocity[day_idx][meal_idx][recipe_idx]

                    # Calculate cognitive and social components
                    r1, r2 = random.random(), random.random()

                    # Cognitive component (difference between pbest and current position)
                    # In discrete space, this is 1 if recipes are different, 0 if same
                    cognitive = 0
                    if pbest[day_idx][meal_idx][recipe_idx] != current_recipe:
                        cognitive = 1

                    # Social component (difference between gbest and current position)
                    social = 0
                    if gbest[day_idx][meal_idx][recipe_idx] != current_recipe:
                        social = 1

                    # Update velocity
                    new_v = (self.w * current_v) + \
                            (self.c1 * r1 * cognitive) + \
                            (self.c2 * r2 * social)

                    # Clamp velocity to [0, 1] as it represents probability
                    new_v = max(0, min(1, new_v))
                    meal_v.append(new_v)
                day_v.append(meal_v)
            new_velocity.append(day_v)

        return new_velocity

    def update_position(self, position, velocity):
        """
        Update position based on velocity

        Args:
            position: Current meal schedule
            velocity: Current velocity (probability of changing recipes)

        Returns:
            Updated position (meal schedule)
        """
        new_position = []

        for day_idx, day_meals in enumerate(position):
            new_day = []
            for meal_idx, meal_recipes in enumerate(day_meals):
                new_meal = []
                for recipe_idx, recipe_id in enumerate(meal_recipes):
                    # Velocity determines probability of changing the recipe
                    if random.random() < velocity[day_idx][meal_idx][recipe_idx]:
                        # Choose a new random recipe
                        new_recipe = random.choice(self.meal_ids)
                        # Ensure we don't pick the same recipe
                        while new_recipe == recipe_id:
                            new_recipe = random.choice(self.meal_ids)
                        new_meal.append(new_recipe)
                    else:
                        # Keep the same recipe
                        new_meal.append(recipe_id)
                new_day.append(new_meal)
            new_position.append(new_day)

        return new_position

    def optimize(self):
        """
        Run the PSO algorithm to find optimal meal schedule

        Returns:
            Tuple of (best meal schedule, nutritional value, fitness score)
        """
        with mlflow.start_run(nested=True, run_name="PSO Optimization") as run:
            # Log PSO parameters
            mlflow.log_params({
                "num_particles": self.num_particles,
                "max_iterations": self.max_iterations,
                "inertia": self.w,
                "cognitive": self.c1,
                "social": self.c2,
                "meals_per_day": self.meals_per_day,
                "recipes_per_meal": self.recipes_per_meal
            })
        
            # Initialize swarm
            positions, velocities, pbests, pbest_scores, gbest, gbest_score = self.initialize_swarm()

            # Optimization loop
            for iteration in range(self.max_iterations):
                # Update each particle
                for i in range(self.num_particles):
                    # Update velocity
                    velocities[i] = self.update_velocity(positions, velocities, pbests, gbest, i)

                    # Update position
                    positions[i] = self.update_position(positions[i], velocities[i])

                    # Evaluate fitness
                    fitness = self.fitness_function(positions[i])

                    # Update personal best
                    if fitness < pbest_scores[i]:
                        pbests[i] = positions[i].copy()
                        pbest_scores[i] = fitness

                        # Update global best
                        if fitness < gbest_score:
                            gbest = positions[i].copy()
                            gbest_score = fitness

                # Print progress every 10 iterations
                if (iteration + 1) % 10 == 0 or iteration == 0:
                    avg_nutrition = self.calculate_nutritional_value(gbest)
                    print(f"Iteration {iteration + 1}: Best fitness = {gbest_score:.4f}, "
                        f"Calories = {avg_nutrition['calories']:.0f}, "
                        f"Protein = {avg_nutrition['protein']:.1f}g")
                    
                if iteration % 5 == 0:
                    current_nutrition = self.calculate_nutritional_value(gbest)
                    mlflow.log_metrics({
                        "fitness": gbest_score,
                        "calories": current_nutrition['calories'],
                        "protein": current_nutrition['protein'],
                        "variety": self.calculate_meal_variety(gbest)
                    }, step=iteration)

            # Calculate final nutritional value
            nutrition = self.calculate_nutritional_value(gbest)
            mlflow.log_metrics({
                "final_fitness": gbest_score,
                "final_calories": nutrition['calories'],
                "final_protein": nutrition['protein'],
                "final_fat": nutrition['fat'],
                "final_carbs": nutrition['carbohydrates'],
                "final_fiber": nutrition['fiber'],
                "unique_meals": self.calculate_meal_variety(gbest)
            })

            temp_plan_path = "temp_best_plan.json"
            with open(temp_plan_path, 'w') as f:
                json.dump(gbest, f)
            mlflow.log_artifact(temp_plan_path, "meal_plans")
            os.remove(temp_plan_path)

            return gbest, nutrition, gbest_score

    def filter_recipes_by_calorie_target(self):
        """
        Filter recipe pool based on calorie target (PSO best practice)
        """
        target_calories = self.target_metrics['calories']

        if target_calories < 2000:  # Low calorie targets
            # Filter recipes under 250 kcal only
            low_cal_mask = self.meal_data['nutrition'].apply(lambda x: x.get('calories', 0) < 250)
            filtered_data = self.meal_data[low_cal_mask]

            if len(filtered_data) < 10:  # Fallback if too few recipes
                # Use recipes under 300 kcal
                fallback_mask = self.meal_data['nutrition'].apply(lambda x: x.get('calories', 0) < 300)
                filtered_data = self.meal_data[fallback_mask]

        elif target_calories > 3200:  # High calorie targets
            # Allow all recipes, prioritize high-calorie ones
            filtered_data = self.meal_data

        else:  # Medium targets
            # Use recipes 150-400 kcal range
            med_cal_mask = self.meal_data['nutrition'].apply(
                lambda x: 150 <= x.get('calories', 0) <= 400
            )
            filtered_data = self.meal_data[med_cal_mask]

        # Update filtered meal data and IDs
        self.filtered_meal_data = filtered_data
        self.meal_ids = filtered_data['ID'].unique().tolist()

        print(f"Filtered to {len(filtered_data)} recipes for target {target_calories} kcal")

    def generate_meal_plan(self):
        """
        Generate an optimized meal plan

        Returns:
            Dictionary containing meal plan details and nutritional information
        """
        # Run optimization
        best_schedule, nutrition, score = self.optimize()

        # Format the meal plan
        meal_plan = []

        for day_idx, day_meals in enumerate(best_schedule):
            day_plan = {
                "day": day_idx + 1,
                "meals": []
            }

            daily_nutrition = {
                'calories': 0,
                'protein': 0,
                'fat': 0,
                'carbohydrates': 0,
                'fiber': 0,
                'calcium': 0
            }

            for meal_idx, meal_recipes in enumerate(day_meals):
                meal_info = {
                    "meal_number": meal_idx + 1,
                    "recipes": []
                }

                meal_nutrition = {
                    'calories': 0,
                    'protein': 0,
                    'fat': 0,
                    'carbohydrates': 0,
                    'fiber': 0,
                    # 'calcium': 0
                }

                for recipe_id in meal_recipes:
                    meal_record = self.filtered_meal_data[self.filtered_meal_data['ID'] == str(recipe_id)]

                    if not meal_record.empty:
                        # Get the first matching record
                        record = meal_record.iloc[0]

                        print(f"Recipe ID: {recipe_id}")
                        print(f"Has Ingredients_Enriched: {'Ingredients_Enriched' in record}")
                        if 'Ingredients_Enriched' in record:
                            ingredients = record['Ingredients_Enriched']
                            total_calcium = sum(float(ing.get('calcium', 0)) for ing in ingredients)
                            print(f"Calculated calcium: {total_calcium}")
                        print("---")

                        # Determine which nutrition data to use
                        if 'Adjusted_Total_Nutrition' in record:
                            recipe_nutrition = record['Adjusted_Total_Nutrition']
                        else:
                            recipe_nutrition = record['nutrition']

                        # Calculate calcium from Ingredients_Enriched
                        recipe_calcium = 0
                        if 'Ingredients_Enriched' in record:
                            ingredients = record['Ingredients_Enriched']
                            recipe_calcium = sum(float(ing.get('calcium', 0)) for ing in ingredients)

                        recipe_info = {
                            "meal_id": recipe_id,
                            "title": record['Title'],
                            "nutrition": {
                                "calories": recipe_nutrition.get('calories', 0),
                                "protein": recipe_nutrition.get('protein', 0),
                                "fat": recipe_nutrition.get('fat', 0),
                                "carbohydrates": recipe_nutrition.get('carbohydrates', 0),
                                "fiber": recipe_nutrition.get('fiber', 0),
                                # "calcium": recipe_calcium  # Use calculated calcium
                            }
                        }

                        meal_nutrition['calories'] += recipe_nutrition.get('calories', 0)
                        meal_nutrition['protein'] += recipe_nutrition.get('protein', 0)
                        meal_nutrition['fat'] += recipe_nutrition.get('fat', 0)
                        meal_nutrition['carbohydrates'] += recipe_nutrition.get('carbohydrates', 0)
                        meal_nutrition['fiber'] += recipe_nutrition.get('fiber', 0)
                        # meal_nutrition['calcium'] += recipe_calcium

                        meal_info["recipes"].append(recipe_info)

                # Add meal nutrition to the meal info
                meal_info["meal_nutrition"] = meal_nutrition

                # Add to daily nutrition totals
                daily_nutrition['calories'] += meal_nutrition['calories']
                daily_nutrition['protein'] += meal_nutrition['protein']
                daily_nutrition['fat'] += meal_nutrition['fat']
                daily_nutrition['carbohydrates'] += meal_nutrition['carbohydrates']
                daily_nutrition['fiber'] += meal_nutrition['fiber']
                # daily_nutrition['calcium'] += meal_nutrition['calcium']

                day_plan["meals"].append(meal_info)

            day_plan["daily_nutrition"] = daily_nutrition
            meal_plan.append(day_plan)

        result = {
            "meal_plan": meal_plan,
            "average_daily_nutrition": nutrition,
            "target_nutrition": self.target_metrics,
            "fitness_score": score
        }

        return result


class MealScheduler:
    def __init__(self, model_dir='models/'):
        """
        Initialize the meal scheduler

        Args:
            model_dir: Directory containing model artifacts
        """
        self.model_dir = model_dir
        self.cbf_recommender = CBFRecommender(model_dir)
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("Meal Planning")

    def create_user_preferences(self, excluded_ingredients=None, dietary_type=None,
                                min_nutrition=None, max_nutrition=None):
        """
        Create user preferences object for filtering recipes

        Args:
            excluded_ingredients: List of ingredients to exclude
            dietary_type: Dietary preference (vegetarian, vegan, pescatarian)
            min_nutrition: Dict with minimum nutrition values
            max_nutrition: Dict with maximum nutrition values

        Returns:
            UserPreferences object
        """
        return UserPreferences(
            excluded_ingredients=excluded_ingredients,
            dietary_type=dietary_type,
            min_nutrition=min_nutrition,
            max_nutrition=max_nutrition
        )

    def generate_meal_plan(self, age, gender, weight, height, activity_level,
                           meals_per_day=3, recipes_per_meal=3, goal='maintain', excluded_ingredients=None,
                           dietary_type=None, min_nutrition=None, max_nutrition=None):
        """
        Generate an optimized meal plan based on user profile and preferences

        Args:
            age: User's age in years
            gender: User's gender ('male' or 'female')
            weight: User's weight in kg
            height: User's height in cm
            activity_level: Physical activity level
            meals_per_day: Number of meals per day (1-4)
            recipes_per_meal: Number of recipes to combine per meal (1-5)
            goal: Weight goal ('lose', 'maintain', 'gain')
            excluded_ingredients: List of ingredients to exclude
            dietary_type: Dietary preference
            min_nutrition: Dict with minimum nutrition values
            max_nutrition: Dict with maximum nutrition values

        Returns:
            Dictionary containing the meal plan
        """
        with mlflow.start_run(nested=True, run_name="PSO Optimization"):
            try:
                # Create user preferences
                user_preferences = self.create_user_preferences(
                    excluded_ingredients=excluded_ingredients,
                    dietary_type=dietary_type,
                    min_nutrition=min_nutrition,
                    max_nutrition=max_nutrition
                )

                # Initialize PSO optimizer
                self.pso_optimizer = ParticleSwarmOptimizer(
                    num_particles=30,
                    num_days=7,
                    meals_per_day=meals_per_day,
                    recipes_per_meal=recipes_per_meal,
                    max_iterations=50
                )

                # Set user requirements for the optimizer
                self.pso_optimizer.set_user_requirements(
                    age=age,
                    gender=gender,
                    weight=weight,
                    height=height,
                    activity_level=activity_level,
                    goal=goal,
                    user_preferences=user_preferences
                )

                # Store the filtered meal data for later use
                self.filtered_meal_data = self.pso_optimizer.filtered_meal_data
                self.target_metrics = self.pso_optimizer.target_metrics

                # Run optimization
                best_schedule, nutrition, score = self.pso_optimizer.optimize()

                # Log basic metrics
                mlflow.log_metrics({
                    "final_calories": nutrition['calories'],
                    "final_protein": nutrition['protein'],
                    "fitness_score": score
                })

                # Format the meal plan
                meal_plan = []

                for day_idx, day_meals in enumerate(best_schedule):
                    day_plan = {
                        "day": day_idx + 1,
                        "meals": []
                    }

                    daily_nutrition = {
                        'calories': 0,
                        'protein': 0,
                        'fat': 0,
                        'carbohydrates': 0,
                        'fiber': 0,
                        # 'calcium': 0
                    }

                    for meal_idx, meal_recipes in enumerate(day_meals):
                        meal_info = {
                            "meal_number": meal_idx + 1,
                            "recipes": []
                        }

                        meal_nutrition = {
                            'calories': 0,
                            'protein': 0,
                            'fat': 0,
                            'carbohydrates': 0,
                            'fiber': 0,
                            # 'calcium': 0
                        }

                        for recipe_id in meal_recipes:
                            meal_record = self.filtered_meal_data[self.filtered_meal_data['ID'] == str(recipe_id)]

                            if not meal_record.empty:
                                recipe_info = {
                                    "meal_id": recipe_id,
                                    "title": meal_record.iloc[0]['Title'],
                                    "nutrition": meal_record.iloc[0]['nutrition']
                                }

                                # Add to meal nutrition totals
                                nutrition = meal_record.iloc[0]['nutrition']
                                meal_nutrition['calories'] += nutrition.get('calories', 0)
                                meal_nutrition['protein'] += nutrition.get('protein', 0)
                                meal_nutrition['fat'] += nutrition.get('fat', 0)
                                meal_nutrition['carbohydrates'] += nutrition.get('carbohydrates', 0)
                                meal_nutrition['fiber'] += nutrition.get('fiber', 0)
                                # meal_nutrition['calcium'] += nutrition.get('calcium', 0)

                                meal_info["recipes"].append(recipe_info)

                        # Add meal nutrition to the meal info
                        meal_info["meal_nutrition"] = meal_nutrition

                        # Add to daily nutrition totals
                        daily_nutrition['calories'] += meal_nutrition['calories']
                        daily_nutrition['protein'] += meal_nutrition['protein']
                        daily_nutrition['fat'] += meal_nutrition['fat']
                        daily_nutrition['carbohydrates'] += meal_nutrition['carbohydrates']
                        daily_nutrition['fiber'] += meal_nutrition['fiber']
                        # daily_nutrition['calcium'] += meal_nutrition['calcium']

                        day_plan["meals"].append(meal_info)

                    day_plan["daily_nutrition"] = daily_nutrition
                    meal_plan.append(day_plan)

                result = {
                    "meal_plan": meal_plan,
                    "average_daily_nutrition": nutrition,
                    "target_nutrition": self.target_metrics,
                    "fitness_score": score
                }

                # Simpan sementara hasil untuk logging
                temp_path = os.path.abspath("temp_meal_plan.json")
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                
                with open(temp_path, 'w') as f:
                    json.dump(result, f, indent=2)
                    
                mlflow.log_artifact(temp_path, "optimization_results")
                os.remove(temp_path)  # Bersihkan file temporary

                return result

            except Exception as e:
                mlflow.log_param("error", str(e))
                raise

    def save_meal_plan(self, meal_plan, output_file='output/meal_plan.json'):
        """
        Save meal plan to a JSON file

        Args:
            meal_plan: Meal plan dictionary
            output_file: Path to output file
        """
        # Ensure output directory exists
        output_file = os.path.abspath(output_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        if mlflow.active_run():
            mlflow.log_artifact(output_file, "meal_plans")
            mlflow.log_dict(meal_plan, "meal_plan_details.json")

        # Convert any numpy values to Python types for JSON serialization
        def convert_numpy_to_python(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_to_python(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_python(item) for item in obj]
            else:
                return obj

        meal_plan_serializable = convert_numpy_to_python(meal_plan)

        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(meal_plan_serializable, f, indent=2, ensure_ascii=False)

        print(f"Meal plan saved to {output_file}")