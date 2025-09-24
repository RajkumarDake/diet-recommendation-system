"""
Database configuration and models for the AI-Powered Nutrition System
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
import logging

from core.config import settings

logger = logging.getLogger(__name__)

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_pre_ping=True
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

class User(Base):
    """User model for storing user information"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    age = Column(Integer)
    gender = Column(String)
    height = Column(Float)  # cm
    weight = Column(Float)  # kg
    activity_level = Column(String)
    dietary_restrictions = Column(JSON)
    allergies = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class HealthData(Base):
    """Health metrics time series data"""
    __tablename__ = "health_data"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    bmi = Column(Float)
    cholesterol_total = Column(Float)
    cholesterol_hdl = Column(Float)
    cholesterol_ldl = Column(Float)
    blood_pressure_systolic = Column(Integer)
    blood_pressure_diastolic = Column(Integer)
    glucose = Column(Float)
    triglycerides = Column(Float)
    heart_rate = Column(Integer)
    additional_metrics = Column(JSON)  # For extensibility

class GenomicData(Base):
    """Genomic SNP data storage"""
    __tablename__ = "genomic_data"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    snp_id = Column(String, nullable=False, index=True)
    chromosome = Column(String)
    position = Column(Integer)
    genotype = Column(String)
    allele_frequency = Column(Float)
    clinical_significance = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class MentalHealthData(Base):
    """Mental health metrics"""
    __tablename__ = "mental_health_data"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    stress_level = Column(Integer)  # 1-10 scale
    anxiety_score = Column(Integer)  # 1-10 scale
    depression_score = Column(Integer)  # 1-10 scale
    sleep_quality = Column(Integer)  # 1-10 scale
    mood_rating = Column(Integer)  # 1-10 scale
    cognitive_function = Column(Integer)  # 1-10 scale
    social_support = Column(Integer)  # 1-10 scale
    additional_assessments = Column(JSON)

class NutritionRecommendation(Base):
    """Generated nutrition recommendations"""
    __tablename__ = "nutrition_recommendations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    recommendation_type = Column(String)  # 'daily', 'weekly', 'meal_plan'
    generated_at = Column(DateTime, default=datetime.utcnow)
    valid_until = Column(DateTime)
    
    # AI Model predictions
    lstm_predictions = Column(JSON)
    transformer_predictions = Column(JSON)
    fusion_predictions = Column(JSON)
    
    # Final recommendations
    nutrition_plan = Column(JSON)
    meal_suggestions = Column(JSON)
    supplement_recommendations = Column(JSON)
    portion_guidance = Column(JSON)
    
    # Confidence and quality metrics
    confidence_score = Column(Float)
    personalization_score = Column(Float)
    
    # User feedback
    user_rating = Column(Integer)  # 1-5 stars
    user_feedback = Column(Text)
    is_active = Column(Boolean, default=True)

class FoodItem(Base):
    """Food database with nutritional information"""
    __tablename__ = "food_items"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    usda_id = Column(String, unique=True, index=True)
    name = Column(String, nullable=False, index=True)
    category = Column(String, index=True)
    
    # Macronutrients per 100g
    calories = Column(Float)
    protein_g = Column(Float)
    carbs_g = Column(Float)
    fat_g = Column(Float)
    fiber_g = Column(Float)
    sugar_g = Column(Float)
    sodium_mg = Column(Float)
    
    # Micronutrients
    vitamins = Column(JSON)
    minerals = Column(JSON)
    
    # Additional properties
    glycemic_index = Column(Integer)
    allergens = Column(JSON)
    sustainability_score = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class MealPlan(Base):
    """Generated meal plans"""
    __tablename__ = "meal_plans"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    date = Column(DateTime, nullable=False)
    meal_type = Column(String)  # breakfast, lunch, dinner, snack
    
    foods = Column(JSON)  # List of food items with portions
    total_calories = Column(Float)
    macronutrient_breakdown = Column(JSON)
    
    generated_at = Column(DateTime, default=datetime.utcnow)
    user_rating = Column(Integer)
    notes = Column(Text)

class ModelTrainingLog(Base):
    """Log of model training sessions"""
    __tablename__ = "model_training_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_type = Column(String, nullable=False)  # lstm, transformer, fusion
    training_started = Column(DateTime, default=datetime.utcnow)
    training_completed = Column(DateTime)
    
    # Training parameters
    epochs = Column(Integer)
    batch_size = Column(Integer)
    learning_rate = Column(Float)
    
    # Training results
    final_loss = Column(Float)
    final_accuracy = Column(Float)
    validation_loss = Column(Float)
    validation_accuracy = Column(Float)
    
    # Model performance metrics
    performance_metrics = Column(JSON)
    training_history = Column(JSON)
    
    # Model file paths
    model_path = Column(String)
    weights_path = Column(String)
    
    status = Column(String, default="started")  # started, completed, failed
    error_message = Column(Text)

class UserFeedback(Base):
    """User feedback on recommendations"""
    __tablename__ = "user_feedback"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    recommendation_id = Column(String, nullable=False)
    feedback_type = Column(String)  # rating, comment, preference_update
    
    rating = Column(Integer)  # 1-5 stars
    comment = Column(Text)
    helpful = Column(Boolean)
    
    # Specific feedback categories
    taste_rating = Column(Integer)
    convenience_rating = Column(Integer)
    health_impact_rating = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class DrugNutrientInteraction(Base):
    """Drug-nutrient interaction database"""
    __tablename__ = "drug_nutrient_interactions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    drug_name = Column(String, nullable=False, index=True)
    nutrient = Column(String, nullable=False, index=True)
    interaction_type = Column(String)  # antagonistic, synergistic, depletion
    severity = Column(String)  # low, moderate, high, severe
    
    description = Column(Text)
    mechanism = Column(Text)
    recommendation = Column(Text)
    
    # Supporting evidence
    evidence_level = Column(String)  # weak, moderate, strong
    references = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)

# Database utility functions
async def init_db():
    """Initialize database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def create_user(db: Session, user_data: Dict[str, Any]) -> User:
    """Create a new user"""
    user = User(**user_data)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

async def get_user(db: Session, user_id: str) -> Optional[User]:
    """Get user by ID"""
    return db.query(User).filter(User.id == user_id).first()

async def create_health_data(db: Session, health_data: Dict[str, Any]) -> HealthData:
    """Create health data entry"""
    health_entry = HealthData(**health_data)
    db.add(health_entry)
    db.commit()
    db.refresh(health_entry)
    return health_entry

async def get_user_health_history(db: Session, user_id: str, limit: int = 100) -> list:
    """Get user's health data history"""
    return db.query(HealthData).filter(
        HealthData.user_id == user_id
    ).order_by(HealthData.timestamp.desc()).limit(limit).all()

async def create_genomic_data(db: Session, genomic_data: Dict[str, Any]) -> GenomicData:
    """Create genomic data entry"""
    genomic_entry = GenomicData(**genomic_data)
    db.add(genomic_entry)
    db.commit()
    db.refresh(genomic_entry)
    return genomic_entry

async def get_user_genomic_data(db: Session, user_id: str) -> list:
    """Get user's genomic data"""
    return db.query(GenomicData).filter(GenomicData.user_id == user_id).all()

async def create_recommendation(db: Session, recommendation_data: Dict[str, Any]) -> NutritionRecommendation:
    """Create nutrition recommendation"""
    recommendation = NutritionRecommendation(**recommendation_data)
    db.add(recommendation)
    db.commit()
    db.refresh(recommendation)
    return recommendation

async def get_user_recommendations(db: Session, user_id: str, active_only: bool = True) -> list:
    """Get user's nutrition recommendations"""
    query = db.query(NutritionRecommendation).filter(NutritionRecommendation.user_id == user_id)
    if active_only:
        query = query.filter(NutritionRecommendation.is_active == True)
    return query.order_by(NutritionRecommendation.generated_at.desc()).all()

async def create_meal_plan(db: Session, meal_plan_data: Dict[str, Any]) -> MealPlan:
    """Create meal plan entry"""
    meal_plan = MealPlan(**meal_plan_data)
    db.add(meal_plan)
    db.commit()
    db.refresh(meal_plan)
    return meal_plan

async def log_model_training(db: Session, training_data: Dict[str, Any]) -> ModelTrainingLog:
    """Log model training session"""
    training_log = ModelTrainingLog(**training_data)
    db.add(training_log)
    db.commit()
    db.refresh(training_log)
    return training_log

async def create_user_feedback(db: Session, feedback_data: Dict[str, Any]) -> UserFeedback:
    """Create user feedback entry"""
    feedback = UserFeedback(**feedback_data)
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback

async def search_foods(db: Session, query: str, category: Optional[str] = None, limit: int = 20) -> list:
    """Search food items in database"""
    db_query = db.query(FoodItem).filter(FoodItem.name.ilike(f"%{query}%"))
    if category:
        db_query = db_query.filter(FoodItem.category == category)
    return db_query.limit(limit).all()

async def get_drug_interactions(db: Session, drug_name: str) -> list:
    """Get drug-nutrient interactions for a specific drug"""
    return db.query(DrugNutrientInteraction).filter(
        DrugNutrientInteraction.drug_name.ilike(f"%{drug_name}%")
    ).all()

# Database health check
async def check_db_health() -> Dict[str, Any]:
    """Check database connectivity and health"""
    try:
        db = SessionLocal()
        # Simple query to test connection
        result = db.execute("SELECT 1").fetchone()
        db.close()
        
        return {
            "status": "healthy",
            "connection": "active",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "connection": "failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
