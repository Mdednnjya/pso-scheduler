from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from fastapi.responses import JSONResponse
import uuid
import json
import asyncio
from datetime import datetime

from api.models.user_models import UserProfile, MealPlanResponse, MealPlanResult
from api.services.meal_service import MealPlanService
from api.services.redis_service import get_redis_client

router = APIRouter(tags=["Meal Planning"])


@router.post("/meal-plan", response_model=MealPlanResponse)
async def create_meal_plan(
        user_profile: UserProfile,
        background_tasks: BackgroundTasks,
        redis_client=Depends(get_redis_client)
):
    """
    Generate optimized meal plan using PSO algorithm

    - **session_id**: Unique identifier for tracking progress
    - **poll_url**: Endpoint to check completion status
    """
    try:
        # Generate unique session ID
        session_id = f"meal_{uuid.uuid4().hex[:8]}"

        # Cache user profile
        await redis_client.setex(
            f"user:{session_id}",
            3600,  # 1 hour TTL
            user_profile.json()
        )

        # Set initial status
        await redis_client.setex(f"status:{session_id}", 3600, "processing")

        # Start background processing
        meal_service = MealPlanService()
        background_tasks.add_task(
            meal_service.process_meal_plan_async,
            session_id,
            user_profile,
            redis_client
        )

        return MealPlanResponse(
            session_id=session_id,
            status="processing",
            message="Your meal plan is being optimized...",
            estimated_time="30-60 seconds",
            poll_url=f"/api/v1/meal-plan/{session_id}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start meal plan generation: {str(e)}"
        )


@router.get("/meal-plan/{session_id}", response_model=MealPlanResult)
async def get_meal_plan(
        session_id: str,
        redis_client=Depends(get_redis_client)
):
    """
    Get meal plan result by session ID

    - **processing**: Still generating
    - **completed**: Meal plan ready
    - **error**: Generation failed
    """
    try:
        # Check status
        status = await redis_client.get(f"status:{session_id}")

        if not status:
            raise HTTPException(
                status_code=404,
                detail="Session not found or expired"
            )

        if status == "completed":
            # Get meal plan
            meal_plan_data = await redis_client.get(f"meal_plan:{session_id}")
            if meal_plan_data:
                meal_plan = json.loads(meal_plan_data)
                return MealPlanResult(
                    session_id=session_id,
                    status="completed",
                    meal_plan=meal_plan
                )

        elif status == "processing":
            return MealPlanResult(
                session_id=session_id,
                status="processing",
                meal_plan=None
            )

        else:  # error status
            error_msg = await redis_client.get(f"error:{session_id}")
            return MealPlanResult(
                session_id=session_id,
                status="error",
                error=error_msg or "Unknown error occurred"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve meal plan: {str(e)}"
        )


# Compatibility endpoint dengan Next.js
@router.post("/schedule/meal")
async def schedule_meal_legacy(
        user: UserProfile,
        background_tasks: BackgroundTasks,
        redis_client=Depends(get_redis_client)
):
    """Legacy endpoint for backward compatibility"""
    return await create_meal_plan(user, background_tasks, redis_client)