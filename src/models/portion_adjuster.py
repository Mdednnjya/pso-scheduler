import logging
import math

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('portion_adjuster')


def calculate_bmr(gender, weight, height, age):
    """
    Calculate Basal Metabolic Rate using Harris-Benedict equation

    Args:
        gender: 'male' or 'female'
        weight: Weight in kg
        height: Height in cm
        age: Age in years

    Returns:
        float: BMR in calories per day
    """
    if gender.lower() == 'male':
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:  # female
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)

    return bmr


def calculate_tdee(bmr, activity_level):
    """
    Calculate Total Daily Energy Expenditure

    Args:
        bmr: Basal Metabolic Rate
        activity_level: Physical activity level

    Returns:
        float: TDEE in calories per day
    """
    # Activity multipliers
    multipliers = {
        'sedentary': 1.2,  # Little or no exercise
        'lightly_active': 1.375,  # Light exercise 1-3 days/week
        'moderately_active': 1.55,  # Moderate exercise 3-5 days/week
        'very_active': 1.725,  # Hard exercise 6-7 days/week
        'extra_active': 1.9  # Very hard exercise & physical job
    }

    # Get multiplier based on activity level
    multiplier = multipliers.get(activity_level.lower(), 1.55)  # Default to moderate if not found

    # Calculate TDEE
    tdee = bmr * multiplier

    return tdee


def adjust_for_goal(tdee, goal):
    """
    Adjust energy needs based on goal

    Args:
        tdee: Total Daily Energy Expenditure
        goal: 'lose', 'maintain', or 'gain'

    Returns:
        float: Adjusted calorie target
    """
    if goal.lower() == 'lose':
        return tdee * 0.85  # 15% calorie deficit
    elif goal.lower() == 'gain':
        return tdee * 1.15  # 15% calorie surplus
    else:  # maintain
        return tdee


def calculate_nutrient_targets(calories, is_teen=False, is_male=True):
    """
    Calculate macronutrient targets based on calories

    Args:
        calories: Daily calorie target
        is_teen: Whether the person is a teenager
        is_male: Whether the person is male

    Returns:
        dict: Target values for each nutrient
    """
    # Protein (higher for teens, and higher for males than females)
    if is_teen:
        protein_pct = 0.25 if is_male else 0.22
    else:
        protein_pct = 0.20 if is_male else 0.18

    # Fat (lower bound of recommended range)
    fat_pct = 0.25

    # Carbs (remainder)
    carb_pct = 1.0 - (protein_pct + fat_pct)

    # Calculate grams of each macronutrient
    protein_grams = (calories * protein_pct) / 4  # 4 calories per gram of protein
    fat_grams = (calories * fat_pct) / 9  # 9 calories per gram of fat
    carb_grams = (calories * carb_pct) / 4  # 4 calories per gram of carbs

    # Fiber recommendation
    if is_teen:
        fiber_grams = 30 if is_male else 25
    else:
        fiber_grams = 38 if is_male else 25

    # Calcium recommendation (mg)
    if is_teen:
        calcium_mg = 1300  # Both teen males and females need 1300mg
    else:
        calcium_mg = 1000  # Adult recommendation

    return {
        'calories': calories,
        'protein': protein_grams,
        'fat': fat_grams,
        'carbohydrates': carb_grams,
        'fiber': fiber_grams,
        # 'calcium': calcium_mg
    }


def get_user_nutrition_targets(user_params):
    """
    Calculate nutrition targets based on user parameters

    Args:
        user_params: Dict with user parameters

    Returns:
        dict: Nutrition targets
    """
    # Extract user parameters
    gender = user_params.get('gender', 'male')
    age = user_params.get('age', 30)
    weight = user_params.get('weight', 70)
    height = user_params.get('height', 170)
    activity_level = user_params.get('activity_level', 'moderately_active')
    goal = user_params.get('goal', 'maintain')

    # Calculate BMR
    bmr = calculate_bmr(gender, weight, height, age)

    # Calculate TDEE
    tdee = calculate_tdee(bmr, activity_level)

    # Adjust for goal
    target_calories = adjust_for_goal(tdee, goal)

    # Calculate nutrient targets
    is_teen = age <= 18
    is_male = gender.lower() == 'male'
    nutrition_targets = calculate_nutrient_targets(target_calories, is_teen, is_male)

    # Log the targets
    logger.info(f"Calculated nutrition targets: {nutrition_targets}")

    return nutrition_targets


def calculate_meal_calories(total_daily_calories, meals_per_day):
    """
    Calculate calories per meal based on distribution pattern

    Args:
        total_daily_calories: Total daily calorie target
        meals_per_day: Number of meals per day

    Returns:
        list: Calories for each meal
    """
    if meals_per_day == 1:
        # One meal - 100% of calories
        return [total_daily_calories]

    elif meals_per_day == 2:
        # Two meals - 40% breakfast, 60% dinner
        return [
            total_daily_calories * 0.4,  # Breakfast
            total_daily_calories * 0.6  # Dinner
        ]

    elif meals_per_day == 3:
        # Three meals - 25% breakfast, 40% lunch, 35% dinner
        return [
            total_daily_calories * 0.25,  # Breakfast
            total_daily_calories * 0.4,  # Lunch
            total_daily_calories * 0.35  # Dinner
        ]

    elif meals_per_day == 4:
        # Four meals - 20% breakfast, 10% snack, 40% lunch, 30% dinner
        return [
            total_daily_calories * 0.2,  # Breakfast
            total_daily_calories * 0.1,  # Snack
            total_daily_calories * 0.4,  # Lunch
            total_daily_calories * 0.3  # Dinner
        ]

    else:
        # Default to equal distribution
        return [total_daily_calories / meals_per_day] * meals_per_day


def adjust_portion_for_user(recipe, user_params, meal_index=0):
    """
    Adjust recipe portion based on user parameters

    Args:
        recipe: Dict with recipe data
        user_params: Dict with user parameters
        meal_index: Index of meal (0=breakfast, 1=lunch, 2=dinner)

    Returns:
        dict: Adjusted recipe
    """
    # Get nutrition targets
    nutrition_targets = get_user_nutrition_targets(user_params)

    # Calculate meal calories
    meals_per_day = user_params.get('meals_per_day', 3)
    meal_calories = calculate_meal_calories(nutrition_targets['calories'], meals_per_day)

    # Determine target calories for this meal
    target_meal_calories = meal_calories[min(meal_index, len(meal_calories) - 1)]

    # Get current recipe calories
    current_recipe_calories = recipe['Total_Nutrition']['calories']
    estimated_portions = recipe.get('Estimated_Portions', 4)

    # Calculate current calories per portion
    current_portion_calories = current_recipe_calories / estimated_portions

    # Calculate portion factor to adjust to target calories
    if current_portion_calories > 0:
        portion_factor = target_meal_calories / current_portion_calories
    else:
        portion_factor = 1.0  # Default if no calories info

    # Cap the portion factor to avoid extreme values
    portion_factor = max(0.25, min(2.0, portion_factor))

    # Create a copy of the recipe
    adjusted_recipe = recipe.copy()

    # Add portion adjustment info
    adjusted_recipe['Portion_Adjustment'] = {
        'original_calories': current_recipe_calories,
        'original_portions': estimated_portions,
        'target_calories': target_meal_calories,
        'portion_factor': portion_factor,
        'adjusted_for': {
            'gender': user_params.get('gender'),
            'age': user_params.get('age'),
            'weight': user_params.get('weight'),
            'height': user_params.get('height'),
            'activity_level': user_params.get('activity_level'),
            'goal': user_params.get('goal')
        }
    }

    # Adjust total nutrition
    adjusted_recipe['Adjusted_Total_Nutrition'] = {
        nutrient: value * portion_factor
        for nutrient, value in recipe['Total_Nutrition'].items()
    }

    # Adjust per-portion nutrition (already per portion, so just scale)
    adjusted_recipe['Adjusted_Per_Portion_Nutrition'] = {
        nutrient: value * portion_factor
        for nutrient, value in recipe.get('Per_Portion_Nutrition', {}).items()
    }

    # Log the adjustment
    logger.info(f"Adjusted recipe {recipe.get('ID')} with portion factor {portion_factor:.2f}")

    return adjusted_recipe


def batch_adjust_recipes(recipes, user_params):
    """
    Adjust portions for a batch of recipes

    Args:
        recipes: List of recipe dictionaries
        user_params: Dict with user parameters

    Returns:
        list: Adjusted recipes
    """
    adjusted_recipes = []

    for recipe in recipes:
        adjusted = adjust_portion_for_user(recipe, user_params)
        adjusted_recipes.append(adjusted)

    return adjusted_recipes