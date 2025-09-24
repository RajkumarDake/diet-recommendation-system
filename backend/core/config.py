"""
Configuration settings for the AI-Powered Nutrition Planning System
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "AI-Powered Personalized Nutrition Planning System"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://127.0.0.1:3000"
    ]
    
    # Database
    DATABASE_URL: str = "sqlite:///./nutrition_system.db"
    
    # Redis (for caching)
    REDIS_URL: str = "redis://localhost:6379"
    
    # AI Models Configuration
    MODEL_PATH: str = "./models"
    LSTM_MODEL_PATH: str = "./models/lstm_health_model.h5"
    TRANSFORMER_MODEL_PATH: str = "./models/transformer_genomic_model"
    FUSION_MODEL_PATH: str = "./models/fusion_model.h5"
    
    # Data Sources
    NHANES_DATA_PATH: str = "./datasets/nhanes_data.csv"
    GENOMIC_DATA_PATH: str = "./datasets/genomic_dataset.csv"
    MENTAL_HEALTH_DATA_PATH: str = "./datasets/mental_health_dataset.csv"
    USDA_NUTRIENT_DATA_PATH: str = "./datasets/usda_nutrients.csv"
    DRUGBANK_DATA_PATH: str = "./datasets/drugbank_interactions.csv"
    
    # External APIs
    DBSNP_API_BASE: str = "https://api.ncbi.nlm.nih.gov/variation/v0"
    USDA_API_KEY: Optional[str] = None
    
    # Model Training Parameters
    LSTM_EPOCHS: int = 50  # Proper training epochs
    TRANSFORMER_EPOCHS: int = 50
    FUSION_EPOCHS: int = 50
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    
    # Feature Engineering
    HEALTH_FEATURES: List[str] = [
        "age", "gender", "height", "weight", "bmi", "cholesterol_total", 
        "cholesterol_hdl", "cholesterol_ldl", "blood_pressure_systolic",
        "blood_pressure_diastolic", "glucose", "triglycerides"
    ]
    
    GENOMIC_FEATURES: List[str] = [
        "rs1801282", "rs1801133", "rs662799", "rs5918", "rs1042713",
        "rs1042714", "rs4680", "rs1799971", "rs16969968", "rs1051730"
    ]
    
    MENTAL_HEALTH_FEATURES: List[str] = [
        "stress_level", "anxiety_score", "depression_score", "sleep_quality",
        "mood_rating", "cognitive_function", "social_support"
    ]
    
    # Recommendation Parameters
    MAX_RECOMMENDATIONS: int = 10
    CONFIDENCE_THRESHOLD: float = 0.7
    UPDATE_FREQUENCY_HOURS: int = 24
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Security
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra environment variables

# Create settings instance
settings = Settings()

# Ensure model directories exist
Path(settings.MODEL_PATH).mkdir(parents=True, exist_ok=True)
Path("./datasets").mkdir(parents=True, exist_ok=True)
Path("./logs").mkdir(parents=True, exist_ok=True)
