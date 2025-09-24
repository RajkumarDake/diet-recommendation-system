"""
Health API Routes
Endpoints for health data management and LSTM model predictions
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import logging
from datetime import datetime

from models.ai_models import ModelManager

logger = logging.getLogger(__name__)
router = APIRouter()

class HealthDataPoint(BaseModel):
    """Single health data point"""
    timestamp: datetime
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

class HealthSequence(BaseModel):
    """Health data sequence for analysis"""
    user_id: str
    data_points: List[HealthDataPoint]

class HealthPrediction(BaseModel):
    """Health prediction response"""
    nutrition_category: Dict[str, Any]
    health_risk: Dict[str, Any]
    metabolic_rate: float
    caloric_needs: float
    recommendations: List[str]

def get_model_manager() -> ModelManager:
    """Dependency to get model manager"""
    from main import app
    return app.state.model_manager

@router.post("/analyze", response_model=HealthPrediction)
async def analyze_health_data(
    health_sequence: HealthSequence,
    model_manager: ModelManager = Depends(get_model_manager)
) -> HealthPrediction:
    """
    Analyze health data sequence using LSTM model
    
    Processes time-series health data to predict nutritional needs,
    health risks, metabolic rate, and caloric requirements.
    """
    try:
        logger.info(f"Analyzing health data for user: {health_sequence.user_id}")
        
        # Validate model availability
        status = await model_manager.get_model_status()
        if not status['lstm_model']['trained']:
            raise HTTPException(
                status_code=503,
                detail="LSTM health model not available. Please train model first."
            )
        
        # Prepare health sequence
        sequence_array = _prepare_health_sequence_array(health_sequence.data_points)
        
        # Get predictions
        predictions = await model_manager.get_health_predictions(sequence_array)
        
        # Generate recommendations based on predictions
        recommendations = _generate_health_recommendations(predictions)
        
        return HealthPrediction(
            nutrition_category=predictions['nutrition_category'],
            health_risk=predictions['health_risk'],
            metabolic_rate=predictions['metabolic_rate'],
            caloric_needs=predictions['caloric_needs'],
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Error analyzing health data: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing health data: {str(e)}")

@router.post("/predict-single")
async def predict_single_health_point(
    health_data: HealthDataPoint,
    model_manager: ModelManager = Depends(get_model_manager)
) -> Dict[str, Any]:
    """
    Predict health metrics for a single data point
    
    Useful for quick health assessments without full sequence data.
    """
    try:
        # Create a sequence by repeating the single point
        repeated_sequence = [health_data] * 30
        sequence_array = _prepare_health_sequence_array(repeated_sequence)
        
        # Get predictions
        predictions = await model_manager.get_health_predictions(sequence_array)
        
        return {
            "predictions": predictions,
            "health_insights": _generate_health_insights(health_data, predictions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error predicting single health point: {e}")
        raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")

@router.get("/metrics/categories")
async def get_health_categories() -> Dict[str, Any]:
    """Get available health categories and their descriptions"""
    return {
        "nutrition_categories": {
            0: "Low Carb, High Protein",
            1: "Mediterranean Diet",
            2: "Low Fat, High Fiber", 
            3: "Ketogenic Diet",
            4: "Plant-Based Diet",
            5: "Balanced Macronutrients",
            6: "Anti-Inflammatory Diet",
            7: "Personalized Micronutrient Focus"
        },
        "health_risk_levels": {
            0: "Very Low Risk",
            1: "Low Risk",
            2: "Moderate Risk",
            3: "High Risk",
            4: "Very High Risk"
        },
        "metabolic_rate_ranges": {
            "low": "< 1500 kcal/day",
            "normal": "1500-2500 kcal/day",
            "high": "> 2500 kcal/day"
        }
    }

@router.post("/validate")
async def validate_health_data(
    health_data: HealthDataPoint
) -> Dict[str, Any]:
    """
    Validate health data for completeness and accuracy
    
    Checks for missing values, outliers, and data quality issues.
    """
    try:
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "completeness_score": 0.0,
            "quality_score": 0.0
        }
        
        # Calculate BMI if not provided
        if health_data.bmi is None:
            calculated_bmi = health_data.weight / ((health_data.height / 100) ** 2)
            validation_results["calculated_bmi"] = calculated_bmi
            
            if calculated_bmi < 16 or calculated_bmi > 40:
                validation_results["warnings"].append(f"BMI {calculated_bmi:.1f} is outside normal range")
        
        # Check for missing critical values
        critical_fields = ['cholesterol_total', 'blood_pressure_systolic', 'glucose']
        missing_fields = []
        
        for field in critical_fields:
            if getattr(health_data, field) is None:
                missing_fields.append(field)
        
        if missing_fields:
            validation_results["warnings"].append(f"Missing critical fields: {', '.join(missing_fields)}")
        
        # Calculate completeness score
        total_fields = 12
        provided_fields = total_fields - len(missing_fields)
        validation_results["completeness_score"] = provided_fields / total_fields
        
        # Check for outliers
        if health_data.cholesterol_total and health_data.cholesterol_total > 300:
            validation_results["warnings"].append("High cholesterol detected")
        
        if health_data.blood_pressure_systolic and health_data.blood_pressure_systolic > 140:
            validation_results["warnings"].append("High blood pressure detected")
        
        if health_data.glucose and health_data.glucose > 126:
            validation_results["warnings"].append("Elevated glucose levels detected")
        
        # Calculate quality score
        validation_results["quality_score"] = max(0.0, 1.0 - (len(validation_results["warnings"]) * 0.1))
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Error validating health data: {e}")
        raise HTTPException(status_code=500, detail=f"Error validating data: {str(e)}")

@router.get("/trends/{user_id}")
async def get_health_trends(
    user_id: str,
    days: int = 30
) -> Dict[str, Any]:
    """
    Get health trends for a user over specified time period
    
    This would typically query a database for historical health data.
    """
    try:
        # Placeholder implementation - would query actual database
        return {
            "user_id": user_id,
            "period_days": days,
            "trends": {
                "weight": {
                    "trend": "stable",
                    "change_percent": 0.5,
                    "direction": "increasing"
                },
                "bmi": {
                    "trend": "stable", 
                    "change_percent": 0.3,
                    "direction": "stable"
                },
                "blood_pressure": {
                    "systolic_trend": "decreasing",
                    "diastolic_trend": "stable",
                    "improvement": True
                },
                "cholesterol": {
                    "total_trend": "decreasing",
                    "hdl_trend": "increasing",
                    "ldl_trend": "decreasing",
                    "improvement": True
                }
            },
            "recommendations": [
                "Continue current exercise routine",
                "Maintain healthy diet",
                "Monitor blood pressure regularly"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting health trends: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting trends: {str(e)}")

# Helper functions
def _prepare_health_sequence_array(data_points: List[HealthDataPoint]) -> np.ndarray:
    """Convert health data points to numpy array for LSTM model"""
    
    sequence_data = []
    
    for point in data_points:
        # Calculate BMI if not provided
        bmi = point.bmi or point.weight / ((point.height / 100) ** 2)
        
        # Create feature vector
        features = [
            point.age,
            1 if point.gender == 'male' else 0,  # Simple gender encoding
            bmi,
            point.cholesterol_total or 200,  # Use defaults for missing values
            point.blood_pressure_systolic or 120,
            point.blood_pressure_diastolic or 80,
            point.glucose or 100,
            point.triglycerides or 150,
            point.cholesterol_hdl or 50,
            point.cholesterol_ldl or 100
        ]
        
        sequence_data.append(features)
    
    # Ensure we have exactly 30 timesteps
    if len(sequence_data) < 30:
        # Pad with the last available data point
        last_point = sequence_data[-1] if sequence_data else [0] * 10
        while len(sequence_data) < 30:
            sequence_data.append(last_point)
    elif len(sequence_data) > 30:
        # Take the last 30 points
        sequence_data = sequence_data[-30:]
    
    return np.array(sequence_data)

def _generate_health_recommendations(predictions: Dict[str, Any]) -> List[str]:
    """Generate health recommendations based on LSTM predictions"""
    
    recommendations = []
    
    # Nutrition category recommendations
    nutrition_class = predictions['nutrition_category']['predicted_class']
    if nutrition_class == 0:  # Low Carb, High Protein
        recommendations.append("Focus on lean proteins and reduce refined carbohydrates")
    elif nutrition_class == 1:  # Mediterranean
        recommendations.append("Incorporate olive oil, fish, and vegetables into your diet")
    elif nutrition_class == 6:  # Anti-inflammatory
        recommendations.append("Include anti-inflammatory foods like turmeric and berries")
    
    # Health risk recommendations
    risk_class = predictions['health_risk']['predicted_class']
    if risk_class >= 3:  # High risk
        recommendations.append("Consult with healthcare provider for comprehensive health assessment")
        recommendations.append("Consider more frequent health monitoring")
    elif risk_class == 2:  # Moderate risk
        recommendations.append("Focus on preventive health measures")
    
    # Metabolic rate recommendations
    metabolic_rate = predictions['metabolic_rate']
    if metabolic_rate < 1500:
        recommendations.append("Consider increasing physical activity to boost metabolism")
    elif metabolic_rate > 2500:
        recommendations.append("Ensure adequate nutrition to support high metabolic rate")
    
    # Caloric needs recommendations
    caloric_needs = predictions['caloric_needs']
    recommendations.append(f"Target approximately {caloric_needs:.0f} calories per day")
    
    return recommendations

def _generate_health_insights(health_data: HealthDataPoint, predictions: Dict[str, Any]) -> Dict[str, Any]:
    """Generate health insights from data and predictions"""
    
    insights = {
        "bmi_status": "",
        "cardiovascular_risk": "",
        "metabolic_health": "",
        "nutrition_focus": ""
    }
    
    # BMI insights
    bmi = health_data.bmi or health_data.weight / ((health_data.height / 100) ** 2)
    if bmi < 18.5:
        insights["bmi_status"] = "Underweight - consider increasing caloric intake"
    elif bmi < 25:
        insights["bmi_status"] = "Normal weight - maintain current lifestyle"
    elif bmi < 30:
        insights["bmi_status"] = "Overweight - consider caloric reduction and exercise"
    else:
        insights["bmi_status"] = "Obese - consult healthcare provider for weight management"
    
    # Cardiovascular risk
    if health_data.blood_pressure_systolic and health_data.blood_pressure_systolic > 140:
        insights["cardiovascular_risk"] = "Elevated blood pressure detected"
    elif health_data.cholesterol_total and health_data.cholesterol_total > 240:
        insights["cardiovascular_risk"] = "High cholesterol levels detected"
    else:
        insights["cardiovascular_risk"] = "Cardiovascular markers within normal range"
    
    # Metabolic health
    if health_data.glucose and health_data.glucose > 126:
        insights["metabolic_health"] = "Elevated glucose - monitor for diabetes risk"
    elif predictions['metabolic_rate'] < 1500:
        insights["metabolic_health"] = "Low metabolic rate - consider increasing activity"
    else:
        insights["metabolic_health"] = "Metabolic health appears normal"
    
    # Nutrition focus
    nutrition_class = predictions['nutrition_category']['predicted_class']
    nutrition_categories = {
        0: "Focus on protein-rich, low-carb foods",
        1: "Mediterranean diet approach recommended",
        2: "Emphasize fiber and limit saturated fats",
        6: "Anti-inflammatory foods recommended"
    }
    insights["nutrition_focus"] = nutrition_categories.get(nutrition_class, "Balanced nutrition approach")
    
    return insights
