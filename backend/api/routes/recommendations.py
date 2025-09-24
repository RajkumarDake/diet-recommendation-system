"""
Recommendations API Routes
Main endpoint for generating personalized nutrition recommendations
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import logging

from models.ai_models import ModelManager
from core.config import settings
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models for request/response
class HealthMetrics(BaseModel):
    """Health metrics input model"""
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(..., pattern="^(male|female|other)$")
    height: float = Field(..., ge=50, le=300)  # cm
    weight: float = Field(..., ge=20, le=500)  # kg
    bmi: Optional[float] = None
    cholesterol_total: Optional[float] = Field(None, ge=0, le=500)
    cholesterol_hdl: Optional[float] = Field(None, ge=0, le=200)
    cholesterol_ldl: Optional[float] = Field(None, ge=0, le=400)
    blood_pressure_systolic: Optional[int] = Field(None, ge=70, le=250)
    blood_pressure_diastolic: Optional[int] = Field(None, ge=40, le=150)
    glucose: Optional[float] = Field(None, ge=50, le=500)
    triglycerides: Optional[float] = Field(None, ge=0, le=1000)

class GenomicData(BaseModel):
    """Genomic SNP data input model"""
    snp_data: Dict[str, str] = Field(..., description="SNP ID to genotype mapping")
    
    class Config:
        schema_extra = {
            "example": {
                "snp_data": {
                    "rs1801282": "0/1",
                    "rs1801133": "1/1", 
                    "rs662799": "0/0",
                    "rs5918": "0/1",
                    "rs1042713": "1/1"
                }
            }
        }

class MentalHealthData(BaseModel):
    """Mental health metrics input model"""
    stress_level: int = Field(..., ge=1, le=10)
    anxiety_score: int = Field(..., ge=1, le=10)
    depression_score: int = Field(..., ge=1, le=10)
    sleep_quality: int = Field(..., ge=1, le=10)
    mood_rating: int = Field(..., ge=1, le=10)
    cognitive_function: int = Field(..., ge=1, le=10)
    social_support: int = Field(..., ge=1, le=10)

class LifestyleData(BaseModel):
    """Lifestyle and activity data"""
    activity_level: str = Field(..., pattern="^(sedentary|light|moderate|active|very_active)$")
    exercise_frequency: int = Field(..., ge=0, le=7)  # days per week
    smoking_status: str = Field(..., pattern="^(never|former|current)$")
    alcohol_consumption: str = Field(..., pattern="^(none|light|moderate|heavy)$")
    dietary_restrictions: List[str] = Field(default=[])
    food_allergies: List[str] = Field(default=[])

class NutritionRequest(BaseModel):
    """Complete nutrition recommendation request"""
    user_id: str = Field(..., min_length=1)
    health_metrics: HealthMetrics
    genomic_data: Optional[GenomicData] = None
    mental_health: MentalHealthData
    lifestyle: LifestyleData
    health_history: Optional[List[HealthMetrics]] = None
    goals: List[str] = Field(default=["general_health"])

class NutritionRecommendation(BaseModel):
    """Nutrition recommendation response"""
    user_id: str
    recommendation_id: str
    timestamp: str
    primary_nutrition_plan: Dict[str, Any]
    meal_timing: Dict[str, Any]
    supplements: List[Dict[str, Any]]
    portion_guidance: Dict[str, Any]
    hydration: Dict[str, Any]
    exercise_nutrition: Dict[str, Any]
    sensitivity_alerts: List[Dict[str, Any]]
    food_recommendations: Dict[str, List[str]]
    personalization_score: float
    confidence_metrics: Dict[str, float]
    next_update_date: str

def get_model_manager() -> ModelManager:
    """Dependency to get model manager from app state"""
    from main import app
    return app.state.model_manager

@router.post("/generate", response_model=NutritionRecommendation)
async def generate_recommendations(
    request: NutritionRequest,
    model_manager: ModelManager = Depends(get_model_manager)
) -> NutritionRecommendation:
    """
    Generate comprehensive personalized nutrition recommendations
    
    This endpoint combines LSTM health analysis, Transformer genomic analysis,
    and fusion model predictions to provide personalized nutrition guidance.
    """
    try:
        logger.info(f"Generating recommendations for user: {request.user_id}")
        
        # Check model status
        status = await model_manager.get_model_status()
        
        # In demo mode, skip training checks
        if not status.get('demo_mode', False):
            if not status['lstm_model']['trained']:
                raise HTTPException(status_code=503, detail="LSTM model not trained. Please run training first.")
            if not status['transformer_model']['trained']:
                raise HTTPException(status_code=503, detail="Transformer model not trained. Please run training first.")
            if not status['fusion_model']['trained']:
                raise HTTPException(status_code=503, detail="Fusion model not trained. Please run training first.")
            logger.info("All models are trained and ready for predictions")
        else:
            logger.info("Running in demo mode with simulated predictions")
        
        # Prepare health sequence data
        health_sequence = _prepare_health_sequence(request.health_metrics, request.health_history)
        
        # Prepare genomic data
        genomic_df = _prepare_genomic_data(request.genomic_data)
        
        # Prepare mental health data
        mental_health_dict = {
            'stress_level': request.mental_health.stress_level,
            'anxiety_score': request.mental_health.anxiety_score,
            'depression_score': request.mental_health.depression_score,
            'sleep_quality': request.mental_health.sleep_quality,
            'mood_rating': request.mental_health.mood_rating,
            'cognitive_function': request.mental_health.cognitive_function,
            'social_support': request.mental_health.social_support
        }
        
        # Generate recommendations using fusion model
        recommendations = await model_manager.get_comprehensive_recommendations(
            health_sequence=health_sequence,
            genomic_data=genomic_df,
            mental_health_data=mental_health_dict
        )
        
        # Apply lifestyle and goal adjustments
        recommendations = _apply_lifestyle_adjustments(recommendations, request.lifestyle, request.goals)
        
        # Generate response
        
        response = NutritionRecommendation(
            user_id=request.user_id,
            recommendation_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            primary_nutrition_plan=recommendations['primary_nutrition_plan'],
            meal_timing=recommendations['meal_timing'],
            supplements=recommendations['supplements'],
            portion_guidance=recommendations['portion_guidance'],
            hydration=recommendations['hydration'],
            exercise_nutrition=recommendations['exercise_nutrition'],
            sensitivity_alerts=recommendations['sensitivity_alerts'],
            food_recommendations=recommendations['food_recommendations'],
            personalization_score=recommendations['personalization_score'],
            confidence_metrics=_extract_confidence_metrics(recommendations),
            next_update_date=(datetime.now() + timedelta(hours=settings.UPDATE_FREQUENCY_HOURS)).isoformat()
        )
        
        logger.info(f"Recommendations generated successfully for user: {request.user_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@router.get("/status")
async def get_recommendation_status(
    model_manager: ModelManager = Depends(get_model_manager)
) -> Dict[str, Any]:
    """Get the status of the recommendation system"""
    try:
        status = await model_manager.get_model_status()
        
        return {
            "system_status": "operational" if status['models_loaded'] else "initializing",
            "models": status,
            "features": {
                "health_analysis": status['lstm_model']['trained'],
                "genomic_analysis": status['transformer_model']['trained'],
                "fusion_recommendations": status['fusion_model']['trained'],
                "real_time_updates": True,
                "multi_modal_integration": True
            },
            "supported_features": settings.HEALTH_FEATURES + settings.GENOMIC_FEATURES + settings.MENTAL_HEALTH_FEATURES
        }
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")

@router.post("/update")
async def update_recommendations(
    user_id: str,
    new_health_data: HealthMetrics,
    model_manager: ModelManager = Depends(get_model_manager)
) -> Dict[str, Any]:
    """
    Update recommendations based on new health data
    
    This endpoint allows for real-time adaptation of recommendations
    as user health metrics evolve.
    """
    try:
        logger.info(f"Updating recommendations for user: {user_id}")
        
        # This would typically involve:
        # 1. Retrieving user's existing data from database
        # 2. Incorporating new health data
        # 3. Re-running predictions with updated data
        # 4. Storing updated recommendations
        
        # For now, return a placeholder response
        return {
            "user_id": user_id,
            "status": "updated",
            "message": "Recommendations updated based on new health data",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error updating recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating recommendations: {str(e)}")

@router.post("/batch")
async def generate_batch_recommendations(
    requests: List[NutritionRequest],
    background_tasks: BackgroundTasks,
    model_manager: ModelManager = Depends(get_model_manager)
) -> Dict[str, Any]:
    """
    Generate recommendations for multiple users in batch
    
    Useful for processing multiple users efficiently.
    """
    try:
        logger.info(f"Processing batch recommendations for {len(requests)} users")
        
        # Add batch processing to background tasks
        background_tasks.add_task(_process_batch_recommendations, requests, model_manager)
        
        return {
            "status": "accepted",
            "message": f"Batch processing started for {len(requests)} users",
            "estimated_completion": "5-10 minutes",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")

# Helper functions
def _prepare_health_sequence(current_health: HealthMetrics, history: Optional[List[HealthMetrics]]) -> np.ndarray:
    """Prepare health sequence for LSTM model"""
    # Calculate BMI if not provided
    if current_health.bmi is None:
        current_health.bmi = current_health.weight / ((current_health.height / 100) ** 2)
    
    # Create feature vector
    features = [
        current_health.age,
        1 if current_health.gender == 'male' else 0,  # Simple encoding
        current_health.bmi,
        current_health.cholesterol_total or 200,  # Default values
        current_health.blood_pressure_systolic or 120,
        current_health.blood_pressure_diastolic or 80,
        current_health.glucose or 100,
        current_health.triglycerides or 150,
        current_health.cholesterol_hdl or 50,
        current_health.cholesterol_ldl or 100
    ]
    
    # If no history, repeat current values to create sequence
    if not history:
        sequence = np.array([features] * 30)  # 30 timesteps
    else:
        # Use history + current (simplified)
        sequence_list = []
        for h in history[-29:]:  # Last 29 historical points
            h_bmi = h.bmi or h.weight / ((h.height / 100) ** 2)
            h_features = [
                h.age, 1 if h.gender == 'male' else 0, h_bmi,
                h.cholesterol_total or 200, h.blood_pressure_systolic or 120,
                h.blood_pressure_diastolic or 80, h.glucose or 100,
                h.triglycerides or 150, h.cholesterol_hdl or 50, h.cholesterol_ldl or 100
            ]
            sequence_list.append(h_features)
        
        # Pad if necessary
        while len(sequence_list) < 29:
            sequence_list.insert(0, features)
        
        sequence_list.append(features)  # Add current
        sequence = np.array(sequence_list)
    
    return sequence

def _prepare_genomic_data(genomic_data: Optional[GenomicData]) -> pd.DataFrame:
    """Prepare genomic data for Transformer model"""
    if not genomic_data:
        # Create dummy genomic data
        dummy_snps = {f'rs{i}': '0/0' for i in range(1000)}
        return pd.DataFrame([dummy_snps])
    
    # Pad SNP data to required size
    snp_dict = genomic_data.snp_data.copy()
    for i in range(len(snp_dict), 1000):
        snp_dict[f'rs{i}'] = '0/0'  # Default genotype
    
    return pd.DataFrame([snp_dict])

def _apply_lifestyle_adjustments(recommendations: Dict[str, Any], 
                                lifestyle: LifestyleData, 
                                goals: List[str]) -> Dict[str, Any]:
    """Apply lifestyle and goal-based adjustments to recommendations"""
    
    # Adjust based on activity level
    if lifestyle.activity_level in ['active', 'very_active']:
        recommendations['hydration']['daily_liters'] *= 1.2
        recommendations['primary_nutrition_plan']['description'] += " Increased protein for active lifestyle."
    
    # Adjust for dietary restrictions
    if 'vegetarian' in lifestyle.dietary_restrictions:
        recommendations['food_recommendations']['recommended_foods'] = [
            food for food in recommendations['food_recommendations']['recommended_foods']
            if food not in ['chicken', 'fish', 'meat']
        ]
        recommendations['food_recommendations']['recommended_foods'].extend(['tofu', 'tempeh', 'legumes'])
    
    # Adjust for goals
    if 'weight_loss' in goals:
        recommendations['portion_guidance']['size'] = 'small'
        recommendations['primary_nutrition_plan']['description'] += " Calorie deficit for weight loss."
    
    return recommendations

def _extract_confidence_metrics(recommendations: Dict[str, Any]) -> Dict[str, float]:
    """Extract confidence metrics from recommendations"""
    return {
        'overall_confidence': recommendations.get('personalization_score', 0.0),
        'nutrition_plan_confidence': recommendations['primary_nutrition_plan'].get('confidence', 0.0),
        'meal_timing_confidence': recommendations['meal_timing'].get('confidence', 0.0),
        'supplement_confidence': np.mean([s.get('confidence', 0.0) for s in recommendations['supplements']]) if recommendations['supplements'] else 0.0
    }

async def _process_batch_recommendations(requests: List[NutritionRequest], model_manager: ModelManager):
    """Background task for processing batch recommendations"""
    logger.info(f"Processing {len(requests)} recommendations in background")
    
    for request in requests:
        try:
            # Process each request (simplified)
            logger.info(f"Processing recommendation for user: {request.user_id}")
            # In real implementation, would generate and store recommendations
            
        except Exception as e:
            logger.error(f"Error processing recommendation for user {request.user_id}: {e}")
    
    logger.info("Batch processing completed")
