import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.pso_meal_scheduler import MealScheduler
from api.models.user_models import UserProfile

logger = logging.getLogger(__name__)


class MealPlanService:
    def __init__(self):
        self.scheduler = MealScheduler()

    async def process_meal_plan_async(
            self,
            session_id: str,
            user_profile: UserProfile,
            redis_client
    ):
        """Background task untuk generate meal plan"""
        try:
            # Set processing start time
            await redis_client.setex(f"started:{session_id}", 3600, "true")

            # Run PSO optimization dengan timeout
            meal_plan = await asyncio.wait_for(
                self._generate_meal_plan(user_profile),
                timeout=60.0  # 1 minute timeout
            )

            # Cache result
            await redis_client.setex(
                f"meal_plan:{session_id}",
                3600,
                json.dumps(meal_plan, ensure_ascii=False)
            )

            # Update status
            await redis_client.setex(f"status:{session_id}", 3600, "completed")

            logger.info(f"✅ Meal plan completed for session: {session_id}")

        except asyncio.TimeoutError:
            # Fallback ke sample meal plan
            fallback_plan = self._get_fallback_meal_plan(user_profile)
            await redis_client.setex(
                f"meal_plan:{session_id}",
                3600,
                json.dumps(fallback_plan, ensure_ascii=False)
            )
            await redis_client.setex(f"status:{session_id}", 3600, "completed")
            logger.warning(f"⚠️ Timeout - using fallback for session: {session_id}")

        except Exception as e:
            # Error handling
            error_msg = f"Generation failed: {str(e)}"
            await redis_client.setex(f"error:{session_id}", 3600, error_msg)
            await redis_client.setex(f"status:{session_id}", 3600, "error")
            logger.error(f"❌ Error for session {session_id}: {error_msg}")

    async def _generate_meal_plan(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Generate meal plan using existing PSO scheduler"""
        loop = asyncio.get_event_loop()

        meal_plan = await loop.run_in_executor(
            None,
            self.scheduler.generate_meal_plan,
            user_profile.age,
            user_profile.gender.value,
            user_profile.weight,
            user_profile.height,
            user_profile.activity_level.value,
            user_profile.meals_per_day,
            user_profile.recipes_per_meal,
            user_profile.goal.value,
            user_profile.exclude,
            user_profile.diet_type.value if user_profile.diet_type else None
        )

        # FIX: Recalculate average daily nutrition correctly
        if 'meal_plan' in meal_plan:
            daily_totals = []
            for day in meal_plan['meal_plan']:
                daily_totals.append(day['daily_nutrition'])

            # Calculate correct average
            avg_nutrition = {
                'calories': sum(day['calories'] for day in daily_totals) / len(daily_totals),
                'protein': sum(day['protein'] for day in daily_totals) / len(daily_totals),
                'fat': sum(day['fat'] for day in daily_totals) / len(daily_totals),
                'carbohydrates': sum(day['carbohydrates'] for day in daily_totals) / len(daily_totals),
                'fiber': sum(day['fiber'] for day in daily_totals) / len(daily_totals)
            }

            # Update with correct average
            meal_plan['average_daily_nutrition'] = avg_nutrition

        return meal_plan

    def _get_fallback_meal_plan(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Fallback meal plan untuk demo"""
        return {
            "status": "fallback",
            "message": "Using optimized sample meal plan",
            "meal_plan": [
                {
                    "day": 1,
                    "meals": [
                        {
                            "meal_number": 1,
                            "recipes": [
                                {
                                    "meal_id": "1",
                                    "title": "Healthy Sample Breakfast",
                                    "nutrition": {"calories": 350, "protein": 15}
                                }
                            ]
                        }
                    ]
                }
            ],
            "user_profile": user_profile.dict()
        }