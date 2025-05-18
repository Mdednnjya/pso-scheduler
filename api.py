# Jalankan di terminal:
# python -m venv venv
# venv\Scripts\activate 
# pip install -r requirements.txt
# pip install fastapi uvicorn pydantic numpy pandas scikit-learn matplotlib mlflow joblib
# uvicorn api:app --reload
# buka Swagger UI di web/browser: http://127.0.0.1:8000/docs

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import json
import os
import shutil
from pathlib import Path

app = FastAPI(title="PSO Meal Scheduler API")

# -------------------- Data Models --------------------

class UserProfile(BaseModel):
    age: int = Field(..., description="User age in years", example=25)
    gender: str = Field(..., description="Gender: male or female", example="male")
    weight: float = Field(..., description="Weight in kilograms", example=70.0)
    height: float = Field(..., description="Height in centimeters", example=170.0)
    activity_level: str = Field(..., description="Activity level", example="moderately_active")
    goal: Optional[str] = Field("maintain", description="Goal: lose, maintain, or gain weight", example="maintain")
    meals_per_day: int = Field(3, description="Number of meals per day", example=3)
    recipes_per_meal: int = Field(3, description="Number of recipes per meal", example=2)
    exclude: Optional[List[str]] = Field(None, description="Ingredients to exclude", example=["beef", "pork"])
    diet_type: Optional[str] = Field(None, description="Diet type", example="pescatarian")

    class Config:
        schema_extra = {
            "example": {
                "age": 25,
                "gender": "male",
                "weight": 70.0,
                "height": 170.0,
                "activity_level": "moderately_active",
                "goal": "maintain",
                "meals_per_day": 3,
                "recipes_per_meal": 2,
                "exclude": ["beef", "pork"],
                "diet_type": "pescatarian"
            }
        }

class PreferenceRequest(BaseModel):
    base_id: List[str] = Field(..., description="List of base recipe IDs", example=["1", "2"])
    exclude: Optional[List[str]] = Field(None, description="Ingredients to exclude", example=["beef"])
    diet_type: Optional[str] = Field(None, description="Diet type", example="vegetarian")
    min_protein: Optional[float] = Field(None, description="Minimum protein", example=20.0)
    max_calories: Optional[float] = Field(None, description="Maximum calories", example=500.0)
    count: int = Field(5, description="Number of recommendations", example=5)

    class Config:
        schema_extra = {
            "example": {
                "base_id": ["1", "2"],
                "exclude": ["beef"],
                "diet_type": "vegetarian",
                "min_protein": 20.0,
                "max_calories": 500.0,
                "count": 5
            }
        }

class TuneRequest(BaseModel):
    meals_per_day: List[int] = Field(..., description="Options for meals per day", example=[2, 3])
    recipes_per_meal: List[int] = Field(..., description="Options for recipes per meal", example=[2, 3])
    user_profile: UserProfile
    preferences: Optional[Dict] = Field(default_factory=dict, description="Preferences as key-value pairs", example={"diet_type": "vegetarian"})

class StatusResponse(BaseModel):
    status: str
    output_file: Optional[str] = None

class BasicStatusResponse(BaseModel):
    status: str

class TuningResponse(BaseModel):
    status: str
    results_dir: Optional[str] = None

# -------------------- Helper --------------------

BASE_DIR = Path(__file__).resolve().parent
MEAL_DATA_PATH = BASE_DIR / "models" / "meal_data.json"
OUTPUT_DIR = BASE_DIR / "output_api"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"
TUNING_DIR = DEFAULT_OUTPUT_DIR / "tuning"
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------- Load Meal Data --------------------

@app.get("/meals")
def get_meals():
    if not MEAL_DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="meal_data.json not found")
    with open(MEAL_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

@app.get("/meals/{meal_id}")
def get_meal_by_id(meal_id: str):
    if not MEAL_DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="meal_data.json not found")
    with open(MEAL_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if item.get("ID") == meal_id:
            return item
    raise HTTPException(status_code=404, detail="Meal not found")

# -------------------- Run Preprocessing --------------------

@app.post("/preprocess", response_model=BasicStatusResponse)
def preprocess_data():
    os.system("python script/run_preprocessing.py")
    return {"status": "Preprocessing completed"}

# -------------------- Train CBF --------------------

@app.post("/recommend/cbf", response_model=BasicStatusResponse)
def run_cbf_training():
    os.system(f"python script/run_cbf_pipeline.py --input {MEAL_DATA_PATH}")
    return {"status": "CBF model trained"}

# -------------------- Recommend by Preference --------------------

@app.post("/recommend/preference", response_model=StatusResponse)
def recommend_by_preference(req: PreferenceRequest):
    base_ids = " ".join(req.base_id)
    args = f"--base-id {base_ids}"
    if req.exclude:
        args += " --exclude " + " ".join(req.exclude)
    if req.diet_type:
        args += f" --diet-type {req.diet_type}"
    if req.min_protein:
        args += f" --min-protein {req.min_protein}"
    if req.max_calories:
        args += f" --max-calories {req.max_calories}"
    args += f" --count {req.count}"
    os.system(f"python script/run_preference_recommender.py {args}")
    src = DEFAULT_OUTPUT_DIR / "preference_recommendations.json"
    dst = OUTPUT_DIR / "preference_recommendations.json"
    if src.exists():
        shutil.copy(src, dst)
    return {"status": "Recommendation generated", "output_file": str(dst)}

# -------------------- Schedule Meal Plan --------------------

@app.post("/schedule/meal", response_model=StatusResponse)
def schedule_meal(user: UserProfile):
    args = f"--age {user.age} --gender {user.gender} --weight {user.weight} --height {user.height} "
    args += f"--activity-level {user.activity_level} --meals-per-day {user.meals_per_day} "
    args += f"--recipes-per-meal {user.recipes_per_meal} --goal {user.goal} "
    if user.exclude:
        args += "--exclude " + " ".join(user.exclude) + " "
    if user.diet_type:
        args += f"--diet-type {user.diet_type}"
    os.system(f"python script/run_meal_scheduler.py {args}")
    output_file = DEFAULT_OUTPUT_DIR / "meal_plan.json"
    if output_file.exists():
        dst = OUTPUT_DIR / "meal_plan.json"
        shutil.copy(output_file, dst)
        return {"status": "Meal plan scheduled", "output_file": str(dst)}
    return {"status": "Meal plan failed"}

# -------------------- Run PSO Tuning --------------------

@app.post("/schedule/tune", response_model=TuningResponse)
def tune_meal_plan(req: TuneRequest):
    config_file = OUTPUT_DIR / "tune_config.json"
    with open(config_file, "w") as f:
        json.dump(req.dict(), f)
    os.system(f"python script/run_pso_tuner.py --config {config_file}")
    if TUNING_DIR.exists():
        dst_tuning = OUTPUT_DIR / "tuning"
        if dst_tuning.exists():
            shutil.rmtree(dst_tuning)
        shutil.copytree(TUNING_DIR, dst_tuning)
    return {"status": "Tuning completed", "results_dir": str(dst_tuning) if TUNING_DIR.exists() else None}
