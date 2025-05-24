from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class Gender(str, Enum):
    male = "male"
    female = "female"

class ActivityLevel(str, Enum):
    sedentary = "sedentary"
    lightly_active = "lightly_active"
    moderately_active = "moderately_active"
    very_active = "very_active"
    extra_active = "extra_active"

class Goal(str, Enum):
    lose = "lose"
    maintain = "maintain"
    gain = "gain"

class DietType(str, Enum):
    vegetarian = "vegetarian"
    vegan = "vegan"
    pescatarian = "pescatarian"

class UserProfile(BaseModel):
    age: int = Field(..., ge=10, le=100, description="Age in years")
    gender: Gender
    weight: float = Field(..., ge=30, le=200, description="Weight in kg")
    height: float = Field(..., ge=100, le=250, description="Height in cm")
    activity_level: ActivityLevel
    goal: Goal = Goal.maintain
    meals_per_day: int = Field(3, ge=1, le=4)
    recipes_per_meal: int = Field(2, ge=1, le=5)
    exclude: Optional[List[str]] = Field(None, description="Ingredients to exclude")
    diet_type: Optional[DietType] = None

    class Config:
        schema_extra = {
            "example": {
                "age": 25,
                "gender": "male",
                "weight": 70.0,
                "height": 175.0,
                "activity_level": "moderately_active",
                "goal": "maintain",
                "meals_per_day": 3,
                "recipes_per_meal": 2,
                "exclude": ["beef"],
                "diet_type": "vegetarian"
            }
        }

class MealPlanResponse(BaseModel):
    session_id: str
    status: str
    message: str
    estimated_time: Optional[str] = None
    poll_url: Optional[str] = None

class MealPlanResult(BaseModel):
    session_id: str
    status: str
    meal_plan: Optional[dict] = None
    error: Optional[str] = None