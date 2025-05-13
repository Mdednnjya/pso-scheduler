import os

class Config:
    """Configuration for the Flask application"""
    
    # Application directories
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    
    # MLflow configuration
    MLFLOW_TRACKING_URI = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME = "Meal Planning"
    
    # API settings
    DEBUG = True
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'default-secret-key-change-in-production')
    
    # Nutrition calculator defaults
    DEFAULT_ACTIVITY_LEVELS = {
        'sedentary': 1.2,
        'lightly_active': 1.375,
        'moderately_active': 1.55,
        'very_active': 1.725,
        'extra_active': 1.9
    }
    
    # Meal planning defaults
    DEFAULT_DAYS = 7
    DEFAULT_MEALS_PER_DAY = 3
    DEFAULT_RECIPES_PER_MEAL = 1
    
    # PSO algorithm parameters
    PSO_PARTICLES = 30
    PSO_MAX_ITERATIONS = 50
    PSO_INERTIA = 0.7
    PSO_COGNITIVE = 1.5
    PSO_SOCIAL = 1.5


class DevelopmentConfig(Config):
    DEBUG = True
    

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    # In production, use environment variables for sensitive data
    SECRET_KEY = os.environ.get('SECRET_KEY')
    

class TestingConfig(Config):
    TESTING = True
    DEBUG = True
    WTF_CSRF_ENABLED = False