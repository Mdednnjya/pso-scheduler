from fastapi import APIRouter, Depends
from api.services.redis_service import get_redis_client
import os

router = APIRouter(tags=["Health Check"])


@router.get("/health")
async def health_check(redis_client=Depends(get_redis_client)):
    """Health check untuk monitoring"""
    try:
        # Test Redis connection
        await redis_client.ping()
        redis_status = "connected"
    except:
        redis_status = "disconnected"

    # Check model files
    model_files = [
        "models/meal_data.json",
        "models/tfidf_vectorizer.pkl",
        "models/scaler.pkl"
    ]

    model_status = {}
    for file_path in model_files:
        model_status[file_path] = os.path.exists(file_path)

    return {
        "status": "healthy",
        "redis": redis_status,
        "models": model_status,
        "version": "2.0.0-async"
    }