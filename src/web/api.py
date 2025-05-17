#import uvicorn
#from fastapi import FastAPI

#api = FastAPI()

#message = "Hai"

#@api.get("/hello")
#def hello():
#    return {"message": message}

#@api.post("/meal-scheduler")
#def helloName(name: str, age: int):
    # meal_plan, nutrition, targets = panggil run_meal_scheduler dari FILE INI

#    return {"message": "Hello, " + name + "!" + " You are " + str(age) + " years old."}

#@api.post("/preference-recommendation")


#if __name__ == "__main__":
#    uvicorn.run(api, host="0.0.0.0", port=8000)

# === api.py (FastAPI server) ===

import uvicorn
import os
import sys
import json
import uuid
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

from fastapi import FastAPI, HTTPException, Query, Path as PathParam, Body, Depends, File, UploadFile, Form, status
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

# Add project root to the system path for module imports
sys.path.append(str(Path(__file__).parent.parent))

# Import custom modules
from run_cbf_pipeline import train_cbf_model, get_recommendations, get_available_recipes, normalize_nutrition_values
from run_meal_evaluator import run_meal_evaluator
from run_meal_scheduler import run_meal_scheduler
from run_preference_recommender import get_recommendations as get_preference_recommendations
from run_preprocessing import run_preprocessing
from scripts.run_pso_tuner import run_pso_tuner

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Smart Food Recommendation System API",
    description="API for meal planning and food recommendations using Content-Based Filtering and Particle Swarm Optimization",
    version="1.0.0"
)

#==========================================================
# Pydantic Models for Request/Response Validation
#==========================================================

# Models for CBF Pipeline
class NutritionInfo(BaseModel):
    """Model for nutritional information data"""
    calories: float
    protein: float
    fat: float
    carbohydrates: float
    fiber: float

class RecipeRecommendationRequest(BaseModel):
    """Request model for recipe recommendation"""
    recipe_ids: List[str] = Field(..., description="List of recipe IDs to base recommendations on")
    count: int = Field(5, description="Number of recommendations to return")
    max_calories: Optional[float] = Field(None, description="Maximum calories limit")
    excluded_ingredients: Optional[List[str]] = Field(None, description="List of ingredients to exclude")
    dietary_type: Optional[str] = Field(None, description="Dietary preference (vegetarian, vegan, pescatarian)")

class RecipeRecommendation(BaseModel):
    """Response model for recipe recommendation"""
    ID: str
    Title: str
    Nutrition: Dict[str, float]
    Similarity: float

class TrainingRequest(BaseModel):
    """Request model for CBF model training"""
    input_path: str = Field(..., description="Path to the enriched recipe JSON data")
    servings: int = Field(2, description="Number of servings to normalize nutrition values by")
    experiment_name: Optional[str] = Field("CBF Training", description="MLflow experiment name")

class TrainingResponse(BaseModel):
    """Response model for CBF model training"""
    status: str
    message: str
    stats: Dict[str, Any]

class RecipeResponse(BaseModel):
    """Response model for recipe details"""
    ID: str
    Title: str
    nutrition: Dict[str, float]

# Models for Meal Evaluation
class EvaluationRequest(BaseModel):
    """Request model for meal plan evaluation"""
    meal_plan: Dict[str, Any]
    generate_plots: bool = False

class EvaluationResponse(BaseModel):
    """Response model for meal plan evaluation"""
    success: bool
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Models for Meal Planning
class UserProfile(BaseModel):
    """User profile data model for meal planning"""
    age: int = Field(..., gt=0, description="User age in years")
    gender: str = Field(..., description="User gender (male or female)")
    weight: float = Field(..., gt=0, description="User weight in kg")
    height: float = Field(..., gt=0, description="User height in cm")
    activity_level: str = Field(..., description="Activity level (sedentary, lightly_active, moderately_active, very_active, extra_active)")

class MealPlanRequest(BaseModel):
    """Request model for meal plan generation"""
    user_profile: UserProfile
    meals_per_day: int = Field(3, ge=1, le=5, description="Number of meals per day (1-5)")
    recipes_per_meal: int = Field(2, ge=1, le=5, description="Number of recipes per meal (1-5)")
    goal: str = Field("maintain", description="Weight goal (lose, maintain, gain)")
    excluded_ingredients: Optional[List[str]] = Field(None, description="List of ingredients to exclude")
    dietary_type: Optional[str] = Field(None, description="Dietary preference (vegetarian, vegan, pescatarian)")
    use_mlflow: bool = Field(False, description="Whether to track with MLflow")

class MealPlanResponse(BaseModel):
    """Response model for meal plan generation"""
    success: bool
    meal_plan: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None
    output_file: Optional[str] = None
    error: Optional[str] = None

# Models for Preference Recommendation
class PreferenceRecommendationRequest(BaseModel):
    """Request model for preference-based recommendation"""
    base_ids: List[str] = Field([], description="List of recipe IDs to base recommendations on")
    exclude: List[str] = Field([], description="List of ingredients to exclude")
    diet_type: Optional[str] = Field(None, description="Dietary preference (vegetarian, vegan, pescatarian)")
    min_protein: Optional[float] = Field(None, description="Minimum protein content")
    max_calories: Optional[float] = Field(None, description="Maximum calories limit")
    count: int = Field(5, description="Number of recommendations to return")

# Models for PSO Parameter Tuning
class ParameterGridItem(BaseModel):
    """Parameter grid for PSO tuning"""
    meals_per_day: List[int] = Field([2, 3, 4], description="Possible values for meals per day")
    recipes_per_meal: List[int] = Field([1, 2, 3], description="Possible values for recipes per meal")

class TunerRequest(BaseModel):
    """Request model for PSO parameter tuning"""
    user_profile: UserProfile
    excluded_ingredients: Optional[List[str]] = Field(None, description="Ingredients to exclude")
    dietary_type: Optional[str] = Field(None, description="Dietary preference (vegetarian, vegan, pescatarian)")
    param_grid: Optional[ParameterGridItem] = Field(None, description="Parameter grid for search")
    output_dir: Optional[str] = Field("output/tuning", description="Output directory for results")

# Models for Preprocessing
class PreprocessingRequest(BaseModel):
    """Request model for data preprocessing"""
    recipe_path: Optional[str] = Field(None, description="Path to recipe data CSV")
    nutrition_path: Optional[str] = Field(None, description="Path to nutrition data CSV")
    output_dir: str = Field("output", description="Output directory for processed files")
    use_mlflow: bool = Field(True, description="Whether to track with MLflow")
    mlflow_uri: str = Field("http://localhost:5000", description="MLflow tracking URI")
    preprocess_nutrition: bool = Field(True, description="Whether to preprocess nutrition data")
    input_nutrition_path: Optional[str] = Field(None, description="Path to raw nutrition data")
    target_serving: int = Field(2, description="Target serving size for normalization")

class ErrorResponse(BaseModel):
    """Generic error response model"""
    error: str

#==========================================================
# Root Endpoint
#==========================================================

@app.get("/", tags=["Root"])
def read_root():
    """
    Root endpoint that provides API information and available endpoints
    
    Returns:
        Basic API information and list of main endpoints
    """
    return {
        "message": "Welcome to Smart Food Recommendation System API",
        "version": "1.0.0",
        "endpoints_groups": [
            "CBF Pipeline: /train, /recommendations, /recipes",
            "Meal Planning: /generate/meal-plan",
            "Meal Evaluation: /evaluate/meal-plan",
            "Preference Recommendation: /recommendations/preference",
            "PSO Tuning: /tune-parameters",
            "Preprocessing: /preprocessing"
        ]
    }

#==========================================================
# Health Check Endpoint
#==========================================================

@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check endpoint to verify API is operational
    
    Returns:
        Health status information
    """
    return {"status": "healthy", "message": "Smart Food Recommendation System API is operational"}

#==========================================================
# Content-Based Filtering Pipeline Endpoints
#==========================================================

@app.post("/train",
         response_model=Union[TrainingResponse, ErrorResponse],
         tags=["CBF Pipeline"],
         description="Train CBF model on recipe data")
def train_model(request: TrainingRequest):
    """
    Train the Content-Based Filtering model on recipe data
    
    Parameters:
        request: Training request with input path and parameters
        
    Returns:
        Training status and statistics
    """
    try:
        if not os.path.exists(request.input_path):
            return {"error": f"Input file {request.input_path} not found"}
        
        # Train the model
        _, _, stats = train_cbf_model(
            input_json_path=request.input_path,
            servings=request.servings
        )
        
        return {
            "status": "success",
            "message": f"Model trained on {stats['num_recipes']} recipes",
            "stats": stats
        }
    except Exception as e:
        return {"error": f"Training error: {str(e)}"}

@app.post("/recommendations", 
         response_model=Union[List[RecipeRecommendation], ErrorResponse],
         tags=["CBF Pipeline"],
         description="Get recipe recommendations based on input recipe IDs")
def get_recipe_recommendations(request: RecipeRecommendationRequest):
    """
    Get recipe recommendations based on input recipe IDs using CBF model
    
    Parameters:
        request: Recommendation request with recipe IDs and filters
        
    Returns:
        List of recipe recommendations with similarity scores
    """
    try:
        recommendations = get_recommendations(
            recipe_ids=request.recipe_ids,
            n=request.count,
            max_calories=request.max_calories,
            excluded_ingredients=request.excluded_ingredients,
            dietary_type=request.dietary_type
        )
        
        if isinstance(recommendations, dict) and "error" in recommendations:
            return recommendations
        
        return recommendations
    except Exception as e:
        return {"error": f"Recommendation error: {str(e)}"}

@app.get("/recipes", 
        response_model=Union[List[RecipeResponse], ErrorResponse],
        tags=["CBF Pipeline"],
        description="Get list of available recipes")
def list_recipes(count: int = Query(20, description="Number of recipes to return")):
    """
    Get a list of available recipes in the system
    
    Parameters:
        count: Number of recipes to return
        
    Returns:
        List of recipe basic information
    """
    try:
        recipes = get_available_recipes(count=count)
        
        if isinstance(recipes, dict) and "error" in recipes:
            return recipes
        
        return recipes
    except Exception as e:
        return {"error": f"Error fetching recipes: {str(e)}"}

@app.get("/recipes/{recipe_id}", 
        response_model=Union[RecipeResponse, ErrorResponse],
        tags=["CBF Pipeline"],
        description="Get recipe details by ID")
def get_recipe_by_id(recipe_id: str = PathParam(..., description="Recipe ID")):
    """
    Get detailed information about a specific recipe by its ID
    
    Parameters:
        recipe_id: Unique identifier of the recipe
        
    Returns:
        Detailed recipe information including nutrition data
    """
    try:
        if not os.path.exists('models/meal_data.json'):
            return {"error": "Model data not found. Train the model first."}
            
        # Load recipe data
        with open('models/meal_data.json', 'r', encoding='utf-8') as f:
            meal_data = json.load(f)
        
        # Find recipe by ID
        for recipe in meal_data:
            if str(recipe['ID']) == str(recipe_id):
                return recipe
        
        return {"error": f"Recipe with ID {recipe_id} not found"}
    except Exception as e:
        return {"error": f"Error fetching recipe: {str(e)}"}

@app.get("/recommendations/{recipe_id}", 
        response_model=Union[List[RecipeRecommendation], ErrorResponse],
        tags=["CBF Pipeline"],
        description="Get recipe recommendations based on a single recipe ID")
def get_recommendations_by_id(
    recipe_id: str = PathParam(..., description="Recipe ID"),
    count: int = Query(5, description="Number of recommendations to return"),
    max_calories: Optional[float] = Query(None, description="Maximum calories limit"),
    excluded_ingredients: Optional[List[str]] = Query(None, description="List of ingredients to exclude"),
    dietary_type: Optional[str] = Query(None, description="Dietary preference (vegetarian, vegan, pescatarian)")
):
    """
    Get recipe recommendations based on a single recipe ID
    
    Parameters:
        recipe_id: Base recipe ID to find recommendations for
        count: Number of recommendations to return
        max_calories: Optional maximum calorie limit
        excluded_ingredients: Optional ingredients to exclude
        dietary_type: Optional dietary preference filter
        
    Returns:
        List of recipe recommendations with similarity scores
    """
    try:
        recommendations = get_recommendations(
            recipe_ids=[recipe_id],
            n=count,
            max_calories=max_calories,
            excluded_ingredients=excluded_ingredients,
            dietary_type=dietary_type
        )
        
        if isinstance(recommendations, dict) and "error" in recommendations:
            return recommendations
        
        return recommendations
    except Exception as e:
        return {"error": f"Recommendation error: {str(e)}"}

#==========================================================
# Meal Evaluation Endpoints
#==========================================================

@app.post("/evaluate/meal-plan", 
         response_model=EvaluationResponse,
         tags=["Meal Evaluation"],
         description="Evaluate a meal plan provided in the request body")
async def evaluate_meal_plan(request: EvaluationRequest):
    """
    Evaluate a meal plan provided directly in the request body
    
    Parameters:
        request: Evaluation request with meal plan data
        
    Returns:
        Evaluation results with nutritional analysis
    """
    try:
        # Run the evaluation with the provided meal plan data
        result = run_meal_evaluator(
            meal_plan_data=request.meal_plan,
            generate_plots=request.generate_plots
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.post("/evaluate/meal-plan/file",
         response_model=EvaluationResponse,
         tags=["Meal Evaluation"],
         description="Evaluate a meal plan from an uploaded JSON file")
async def evaluate_meal_plan_file(
    file: UploadFile = File(...),
    generate_plots: bool = Form(False)
):
    """
    Evaluate a meal plan from an uploaded JSON file
    
    Parameters:
        file: Uploaded JSON file containing meal plan data
        generate_plots: Whether to generate and include plots in the response
        
    Returns:
        Evaluation results with nutritional analysis
    """
    # Create a temporary file
    temp_dir = tempfile.mkdtemp()
    temp_file_path = Path(temp_dir) / "meal_plan.json"
    
    try:
        # Save the uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run the evaluation with the saved file
        result = run_meal_evaluator(
            meal_plan_file=str(temp_file_path),
            generate_plots=generate_plots
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)

@app.get("/evaluate/sample",
        response_model=EvaluationResponse,
        tags=["Meal Evaluation"],
        description="Evaluate a sample meal plan for testing")
async def evaluate_sample():
    """
    Evaluate a sample meal plan (for testing purposes)
    
    Returns:
        Evaluation results for the sample meal plan
    """
    # Path to a sample meal plan
    sample_file = Path(__file__).parent / "output" / "sample_meal_plan.json"
    
    if not sample_file.exists():
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": "Sample meal plan not found"}
        )
    
    result = run_meal_evaluator(
        meal_plan_file=str(sample_file),
        generate_plots=True
    )
    
    return result

#==========================================================
# Meal Planning Endpoints
#==========================================================

@app.post("/generate/meal-plan",
         response_model=MealPlanResponse,
         tags=["Meal Planning"],
         description="Generate an optimized meal plan based on user parameters")
async def generate_meal_plan(request: MealPlanRequest):
    """
    Generate an optimized meal plan based on user parameters using PSO algorithm
    
    Parameters:
        request: Meal plan generation request with user profile and preferences
        
    Returns:
        Generated meal plan and summary information
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Generate unique filename for the output
        unique_id = str(uuid.uuid4())[:8]
        output_file = str(output_dir / f"meal_plan_{unique_id}.json")
        
        # Extract user profile from request
        user = request.user_profile
        
        # Run the scheduler
        result = run_meal_scheduler(
            age=user.age,
            gender=user.gender,
            weight=user.weight,
            height=user.height,
            activity_level=user.activity_level,
            meals_per_day=request.meals_per_day,
            recipes_per_meal=request.recipes_per_meal,
            goal=request.goal,
            excluded_ingredients=request.excluded_ingredients,
            dietary_type=request.dietary_type,
            output_file=output_file,
            use_mlflow=request.use_mlflow
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Meal plan generation failed: {str(e)}")

@app.post("/generate/meal-plan/quick",
         response_model=MealPlanResponse,
         tags=["Meal Planning"],
         description="Quickly generate a meal plan with minimal parameters")
async def generate_quick_meal_plan(
    age: int = Query(..., description="User age in years"),
    gender: str = Query(..., description="User gender (male or female)"),
    weight: float = Query(..., description="User weight in kg"),
    height: float = Query(..., description="User height in cm"),
    activity_level: str = Query(..., description="Activity level"),
    meals_per_day: int = Query(3, description="Number of meals per day"),
    goal: str = Query("maintain", description="Weight goal"),
    dietary_type: Optional[str] = Query(None, description="Dietary preference")
):
    """
    Simplified endpoint to quickly generate a meal plan with minimal parameters
    
    This is a streamlined version of the meal plan generation endpoint with fewer parameters,
    intended for quick testing or simple use cases.
    
    Parameters:
        age: User age in years
        gender: User gender (male or female)
        weight: User weight in kg
        height: User height in cm
        activity_level: Activity level
        meals_per_day: Number of meals per day
        goal: Weight goal
        dietary_type: Dietary preference
        
    Returns:
        Generated meal plan and summary information
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Generate unique filename for the output
        unique_id = str(uuid.uuid4())[:8]
        output_file = str(output_dir / f"quick_meal_plan_{unique_id}.json")
        
        # Run the scheduler with default values for some parameters
        result = run_meal_scheduler(
            age=age,
            gender=gender,
            weight=weight,
            height=height,
            activity_level=activity_level,
            meals_per_day=meals_per_day,
            recipes_per_meal=2,  # Default to 2 recipes per meal for simplicity
            goal=goal,
            dietary_type=dietary_type,
            output_file=output_file
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick meal plan generation failed: {str(e)}")

@app.post("/generate/analyze-meal-plan",
         response_model=Dict[str, Any],
         tags=["Meal Planning"],
         description="Generate and analyze a meal plan in one request")
async def generate_and_analyze_meal_plan(request: MealPlanRequest):
    """
    Generate a meal plan and immediately analyze its quality
    
    This endpoint combines the meal plan generation and evaluation in one call.
    
    Parameters:
        request: Meal plan generation request with user profile and preferences
        
    Returns:
        Combined generation and evaluation results
    """
    try:
        # First generate the meal plan
        meal_plan_result = await generate_meal_plan(request)
        
        if not meal_plan_result['success']:
            return meal_plan_result
        
        # Then evaluate the meal plan
        evaluation_result = run_meal_evaluator(
            meal_plan_data=meal_plan_result['meal_plan'],
            generate_plots=True
        )
        
        # Combine the results
        combined_result = {
            'success': True,
            'meal_plan': meal_plan_result['meal_plan'],
            'summary': meal_plan_result['summary'],
            'evaluation': evaluation_result['results'] if evaluation_result['success'] else None,
            'output_file': meal_plan_result['output_file']
        }
        
        return combined_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generate and analyze operation failed: {str(e)}")

#==========================================================
# Preference-Based Recommendation Endpoints
#==========================================================

@app.post("/recommendations/preference",
         tags=["Preference Recommendation"],
         description="Get recipe recommendations based on user preferences")
async def preference_recommender_endpoint(request: PreferenceRecommendationRequest):
    """
    Get recipe recommendations based on user preferences
    
    Parameters:
        request: Preference recommendation request with user preferences
        
    Returns:
        List of recipe recommendations matching the preferences
    """
    try:
        # Validate parameters
        if request.diet_type and request.diet_type not in ['vegetarian', 'vegan', 'pescatarian']:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Invalid diet type. Choose from: vegetarian, vegan, pescatarian"
                }
            )
        
        # Get recommendations
        result = get_preference_recommendations(
            base_ids=request.base_ids,
            exclude=request.exclude,
            diet_type=request.diet_type,
            min_protein=request.min_protein,
            max_calories=request.max_calories,
            count=request.count
        )
        
        return result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"An error occurred: {str(e)}"
            }
        )

@app.get("/recipes/filtered",
        tags=["Preference Recommendation"],
        description="Get available recipes that match certain criteria")
async def available_recipes_endpoint(
    exclude: Optional[str] = Query("", description="Comma-separated ingredients to exclude"),
    diet_type: Optional[str] = Query(None, description="Dietary preference"),
    min_protein: Optional[float] = Query(None, description="Minimum protein content"),
    max_calories: Optional[float] = Query(None, description="Maximum calories"),
    limit: int = Query(10, description="Number of recipes to return")
):
    """
    Get available recipes that match certain criteria
    
    Parameters:
        exclude: Comma-separated ingredients to exclude
        diet_type: Dietary preference filter
        min_protein: Minimum protein content
        max_calories: Maximum calories
        limit: Number of recipes to return
        
    Returns:
        List of recipes matching the criteria
    """
    try:
        # Process exclude ingredients
        excluded_list = [item.strip() for item in exclude.split(',')] if exclude else []
        
        # Get available recipes
        result = get_preference_recommendations(
            base_ids=[],  # Empty base_ids to get available recipes
            exclude=excluded_list,
            diet_type=diet_type,
            min_protein=min_protein,
            max_calories=max_calories,
            count=limit
        )
        
        return result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"An error occurred: {str(e)}"
            }
        )

#==========================================================
# PSO Parameter Tuning Endpoints
#==========================================================

@app.post("/tune-parameters",
         tags=["PSO Tuning"],
         description="Tune PSO meal scheduler parameters to find optimal values")
async def tune_parameters(request: TunerRequest = Body(...)):
    """
    Tune PSO meal scheduler parameters to find optimal values
    
    Performs grid search over parameter combinations and evaluates meal plans 
    generated with each parameter set.
    
    Parameters:
        request: Parameter tuning request with user profile and parameter grid
        
    Returns:
        Best parameters and comparison data
    """
    # Convert the model to dict and prepare input for the function
    user_data = {
        "age": request.user_profile.age,
        "gender": request.user_profile.gender,
        "weight": request.user_profile.weight,
        "height": request.user_profile.height,
        "activity_level": request.user_profile.activity_level,
        "excluded_ingredients": request.excluded_ingredients,
        "dietary_type": request.dietary_type,
        "output_dir": request.output_dir
    }
    
    # Add param_grid if provided
    if request.param_grid:
        user_data["param_grid"] = {
            "meals_per_day": request.param_grid.meals_per_day,
            "recipes_per_meal": request.param_grid.recipes_per_meal
        }
    
    # Run the tuner
    result = run_pso_tuner(user_data)
    
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result.get("error", "Unknown error occurred"))
    
    return result

#==========================================================
# Preprocessing Endpoints
#==========================================================

@app.post("/preprocessing/run",
         tags=["Preprocessing"],
         description="Run data preprocessing pipeline")
async def preprocessing_run_endpoint(request: PreprocessingRequest):
    """
    Run data preprocessing pipeline
    
    Parameters:
        request: Preprocessing request with file paths and options
        
    Returns:
        Preprocessing results
    """
    try:
        # Run preprocessing
        result = run_preprocessing(
            recipe_path=request.recipe_path,
            nutrition_path=request.nutrition_path,
            output_dir=request.output_dir,
            use_mlflow=request.use_mlflow,
            mlflow_uri=request.mlflow_uri,
            preprocess_nutrition=request.preprocess_nutrition,
            input_nutrition_path=request.input_nutrition_path,
            target_serving=request.target_serving
        )
        
        return result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"An error occurred: {str(e)}"
            }
        )

@app.post("/preprocessing/upload",
         tags=["Preprocessing"],
         description="Upload files for preprocessing")
async def upload_preprocessing_data(
    recipe_file: Optional[UploadFile] = File(None),
    nutrition_file: Optional[UploadFile] = File(None)
):
    """
    Upload files for preprocessing
    
    Parameters:
        recipe_file: CSV file with recipe data
        nutrition_file: CSV file with nutrition data
        
    Returns:
        Upload status and file paths
    """
    try:
        result = {"status": "success", "uploaded_files": {}}
        
        if not recipe_file and not nutrition_file:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "No files provided"
                }
            )
        
        # Create upload directory if it doesn't exist
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Process recipe file
        if recipe_file and recipe_file.filename:
            # Generate unique filename to prevent overwrites
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{unique_id}_{recipe_file.filename}"
            recipe_path = str(upload_dir / filename)
            
            # Save the file
            with open(recipe_path, "wb") as buffer:
                content = await recipe_file.read()
                buffer.write(content)
                
            result["uploaded_files"]["recipe_path"] = recipe_path
        
        # Process nutrition file
        if nutrition_file and nutrition_file.filename:
            # Generate unique filename to prevent overwrites
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{unique_id}_{nutrition_file.filename}"
            nutrition_path = str(upload_dir / filename)
            
            # Save the file
            with open(nutrition_path, "wb") as buffer:
                content = await nutrition_file.read()
                buffer.write(content)
                
            result["uploaded_files"]["nutrition_path"] = nutrition_path
        
        result["message"] = f"Successfully uploaded {len(result['uploaded_files'])} file(s)"
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"An error occurred during file upload: {str(e)}"
            }
        )