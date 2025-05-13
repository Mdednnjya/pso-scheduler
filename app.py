from flask import Flask, request, jsonify
import json
import os
import pandas as pd
from src.models.user_preferences import UserPreferences
from src.models.pso_meal_scheduler import MealScheduler
from src.models.cbf_recommender import CBFRecommender

app = Flask(__name__)

# Load configuration
app.config.from_object('config.Config')

# Initialize recommender and scheduler
meal_scheduler = MealScheduler(model_dir=app.config['MODEL_DIR'])
cbf_recommender = CBFRecommender(model_dir=app.config['MODEL_DIR'])

@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Food Recommendation API is running'
    })

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Get food recommendations based on input recipes"""
    try:
        data = request.json
        
        # Extract parameters
        recipe_ids = data.get('recipe_ids', [])
        num_recommendations = data.get('num', 5)
        max_calories = data.get('max_calories')
        
        # Extract user preferences if provided
        preferences_data = data.get('preferences', {})
        
        user_preferences = None
        if preferences_data:
            user_preferences = UserPreferences(
                excluded_ingredients=preferences_data.get('excluded_ingredients'),
                dietary_type=preferences_data.get('dietary_type'),
                min_nutrition=preferences_data.get('min_nutrition'),
                max_nutrition=preferences_data.get('max_nutrition')
            )
        
        # Get recommendations
        recommendations = cbf_recommender.recommend(
            recipe_ids=recipe_ids,
            n=num_recommendations,
            max_calories=max_calories,
            user_preferences=user_preferences
        )
        
        # Convert DataFrame to JSON-serializable format
        if not recommendations.empty:
            result = recommendations.to_dict(orient='records')
            for item in result:
                # Ensure all values are JSON serializable
                if 'nutrition' in item and isinstance(item['nutrition'], pd.Series):
                    item['nutrition'] = item['nutrition'].to_dict()
                
            return jsonify({
                'status': 'success',
                'recommendations': result
            })
        else:
            return jsonify({
                'status': 'warning',
                'message': 'No recommendations found for the given criteria',
                'recommendations': []
            })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/meal-plan', methods=['POST'])
def generate_meal_plan():
    """Generate a personalized meal plan"""
    try:
        data = request.json
        
        # User profile parameters
        age = data.get('age')
        gender = data.get('gender')
        weight = data.get('weight')  # kg
        height = data.get('height')  # cm
        activity_level = data.get('activity_level')
        goal = data.get('goal', 'maintain')
        
        # Meal plan configuration
        meals_per_day = data.get('meals_per_day', 3)
        recipes_per_meal = data.get('recipes_per_meal', 1)
        
        # User preferences
        preferences = data.get('preferences', {})
        excluded_ingredients = preferences.get('excluded_ingredients', [])
        dietary_type = preferences.get('dietary_type')
        min_nutrition = preferences.get('min_nutrition', {})
        max_nutrition = preferences.get('max_nutrition', {})
        
        # Validate required parameters
        required_params = ['age', 'gender', 'weight', 'height', 'activity_level']
        missing_params = [param for param in required_params if not data.get(param)]
        
        if missing_params:
            return jsonify({
                'status': 'error',
                'message': f'Missing required parameters: {", ".join(missing_params)}'
            }), 400
        
        # Generate meal plan
        meal_plan = meal_scheduler.generate_meal_plan(
            age=age,
            gender=gender,
            weight=weight,
            height=height,
            activity_level=activity_level,
            meals_per_day=meals_per_day,
            recipes_per_meal=recipes_per_meal,
            goal=goal,
            excluded_ingredients=excluded_ingredients,
            dietary_type=dietary_type,
            min_nutrition=min_nutrition,
            max_nutrition=max_nutrition
        )
        
        # Save meal plan
        output_dir = app.config['OUTPUT_DIR']
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique filename
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"meal_plan_{timestamp}.json"
        output_path = os.path.join(output_dir, filename)
        
        # Save the meal plan
        meal_scheduler.save_meal_plan(meal_plan, output_path)
        
        return jsonify({
            'status': 'success',
            'message': 'Meal plan generated successfully',
            'filename': filename,
            'meal_plan': meal_plan
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/recipes', methods=['GET'])
def get_recipes():
    """Get available recipes with optional filtering"""
    try:
        # Get query parameters
        search_query = request.args.get('search', '')
        limit = int(request.args.get('limit', 10))
        offset = int(request.args.get('offset', 0))
        
        # Get recipes data
        recipes_df = cbf_recommender.meal_data
        
        # Apply search filter if provided
        if search_query:
            search_query = search_query.lower()
            recipes_df = recipes_df[recipes_df['Title'].str.lower().str.contains(search_query)]
        
        # Count total matches
        total_count = len(recipes_df)
        
        # Apply pagination
        recipes_df = recipes_df.iloc[offset:offset+limit]
        
        # Convert to list of dictionaries
        recipes = recipes_df.to_dict(orient='records')
        
        return jsonify({
            'status': 'success',
            'total': total_count,
            'offset': offset,
            'limit': limit,
            'recipes': recipes
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/user-preferences', methods=['POST'])
def test_user_preferences():
    """Test user preferences by returning number of filtered recipes"""
    try:
        data = request.json
        
        # Get user preferences
        excluded_ingredients = data.get('excluded_ingredients', [])
        dietary_type = data.get('dietary_type')
        min_nutrition = data.get('min_nutrition', {})
        max_nutrition = data.get('max_nutrition', {})
        
        # Create user preferences object
        user_prefs = UserPreferences(
            excluded_ingredients=excluded_ingredients,
            dietary_type=dietary_type,
            min_nutrition=min_nutrition,
            max_nutrition=max_nutrition
        )
        
        # Filter recipes based on preferences
        filtered_recipes = user_prefs.filter_recipes(cbf_recommender.meal_data)
        
        # Calculate percentage of recipes filtered
        total_recipes = len(cbf_recommender.meal_data)
        filtered_count = len(filtered_recipes)
        
        # Return sample of filtered recipes
        sample_size = min(5, filtered_count)
        sample_recipes = filtered_recipes.sample(n=sample_size) if filtered_count > 0 else pd.DataFrame()
        
        return jsonify({
            'status': 'success',
            'total_recipes': total_recipes,
            'filtered_count': filtered_count,
            'filter_percentage': (filtered_count / total_recipes) * 100 if total_recipes > 0 else 0,
            'sample_recipes': sample_recipes.to_dict(orient='records') if not sample_recipes.empty else []
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)