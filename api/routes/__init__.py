# api/routes/__init__.py
from .meal_plan import router as meal_plan_router
from .health import router as health_router

__all__ = ["meal_plan_router", "health_router"]