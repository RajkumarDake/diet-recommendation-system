"""
Nutrition API Routes
Endpoints for nutrition data management and food recommendations
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()

class FoodItem(BaseModel):
    """Individual food item with nutritional data"""
    food_id: str
    name: str
    category: str
    calories_per_100g: float = Field(..., ge=0)
    protein_g: float = Field(..., ge=0)
    carbs_g: float = Field(..., ge=0)
    fat_g: float = Field(..., ge=0)
    fiber_g: float = Field(..., ge=0)
    sugar_g: float = Field(..., ge=0)
    sodium_mg: float = Field(..., ge=0)
    vitamins: Dict[str, float] = Field(default={})
    minerals: Dict[str, float] = Field(default={})

class MealPlan(BaseModel):
    """Daily meal plan"""
    date: str
    breakfast: List[FoodItem]
    lunch: List[FoodItem]
    dinner: List[FoodItem]
    snacks: List[FoodItem]
    total_calories: float
    macronutrient_breakdown: Dict[str, float]

class NutrientProfile(BaseModel):
    """User's nutrient requirements and preferences"""
    user_id: str
    daily_calories: int = Field(..., ge=800, le=5000)
    protein_percent: float = Field(..., ge=10, le=40)
    carb_percent: float = Field(..., ge=20, le=70)
    fat_percent: float = Field(..., ge=15, le=50)
    dietary_restrictions: List[str] = Field(default=[])
    allergies: List[str] = Field(default=[])
    preferred_cuisines: List[str] = Field(default=[])
    meal_frequency: int = Field(default=3, ge=1, le=8)

class SupplementRecommendation(BaseModel):
    """Supplement recommendation"""
    supplement_name: str
    dosage: str
    timing: str
    reason: str
    confidence_score: float
    interactions: List[str] = Field(default=[])
    cost_estimate: Optional[float] = None

@router.get("/foods/search")
async def search_foods(
    query: str = Query(..., min_length=2),
    category: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100)
) -> List[FoodItem]:
    """
    Search for foods in the USDA nutrition database
    
    Searches through comprehensive food database with nutritional information
    """
    try:
        # Placeholder food database (would connect to USDA FoodData Central API)
        food_database = [
            FoodItem(
                food_id="usda_001",
                name="Chicken Breast, Skinless",
                category="Protein",
                calories_per_100g=165,
                protein_g=31.0,
                carbs_g=0.0,
                fat_g=3.6,
                fiber_g=0.0,
                sugar_g=0.0,
                sodium_mg=74,
                vitamins={"B6": 0.9, "B12": 0.3, "Niacin": 14.8},
                minerals={"Phosphorus": 228, "Selenium": 27.6}
            ),
            FoodItem(
                food_id="usda_002",
                name="Salmon, Atlantic",
                category="Protein",
                calories_per_100g=208,
                protein_g=25.4,
                carbs_g=0.0,
                fat_g=12.4,
                fiber_g=0.0,
                sugar_g=0.0,
                sodium_mg=59,
                vitamins={"D": 11.0, "B12": 3.2, "B6": 0.8},
                minerals={"Omega3": 2.3, "Selenium": 36.5}
            ),
            FoodItem(
                food_id="usda_003",
                name="Spinach, Raw",
                category="Vegetables",
                calories_per_100g=23,
                protein_g=2.9,
                carbs_g=3.6,
                fat_g=0.4,
                fiber_g=2.2,
                sugar_g=0.4,
                sodium_mg=79,
                vitamins={"K": 483.0, "A": 469.0, "Folate": 194.0},
                minerals={"Iron": 2.7, "Magnesium": 79}
            ),
            FoodItem(
                food_id="usda_004",
                name="Quinoa, Cooked",
                category="Grains",
                calories_per_100g=120,
                protein_g=4.4,
                carbs_g=21.3,
                fat_g=1.9,
                fiber_g=2.8,
                sugar_g=0.9,
                sodium_mg=7,
                vitamins={"Folate": 42.0, "E": 0.6},
                minerals={"Magnesium": 64, "Phosphorus": 152}
            )
        ]
        
        # Filter by query and category
        filtered_foods = []
        for food in food_database:
            if query.lower() in food.name.lower():
                if category is None or food.category.lower() == category.lower():
                    filtered_foods.append(food)
        
        return filtered_foods[:limit]
        
    except Exception as e:
        logger.error(f"Error searching foods: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching foods: {str(e)}")

@router.post("/meal-plan/generate")
async def generate_meal_plan(
    nutrient_profile: NutrientProfile,
    days: int = Query(7, ge=1, le=30)
) -> List[MealPlan]:
    """
    Generate personalized meal plan based on nutrient profile
    
    Creates optimized meal plans considering dietary restrictions,
    preferences, and nutritional requirements.
    """
    try:
        logger.info(f"Generating {days}-day meal plan for user: {nutrient_profile.user_id}")
        
        meal_plans = []
        
        # Get available foods (filtered by restrictions)
        available_foods = await _get_filtered_foods(nutrient_profile)
        
        for day in range(days):
            date = datetime.now().strftime(f"%Y-%m-%d")
            
            # Generate meals for the day
            breakfast = _generate_meal(available_foods, "breakfast", nutrient_profile)
            lunch = _generate_meal(available_foods, "lunch", nutrient_profile)
            dinner = _generate_meal(available_foods, "dinner", nutrient_profile)
            snacks = _generate_meal(available_foods, "snack", nutrient_profile)
            
            # Calculate totals
            all_foods = breakfast + lunch + dinner + snacks
            total_calories = sum(food.calories_per_100g for food in all_foods)
            
            macronutrient_breakdown = {
                "protein_percent": sum(food.protein_g for food in all_foods) * 4 / total_calories * 100,
                "carb_percent": sum(food.carbs_g for food in all_foods) * 4 / total_calories * 100,
                "fat_percent": sum(food.fat_g for food in all_foods) * 9 / total_calories * 100
            }
            
            meal_plan = MealPlan(
                date=date,
                breakfast=breakfast,
                lunch=lunch,
                dinner=dinner,
                snacks=snacks,
                total_calories=total_calories,
                macronutrient_breakdown=macronutrient_breakdown
            )
            
            meal_plans.append(meal_plan)
        
        return meal_plans
        
    except Exception as e:
        logger.error(f"Error generating meal plan: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating meal plan: {str(e)}")

@router.get("/nutrients/requirements/{user_id}")
async def get_nutrient_requirements(
    user_id: str,
    age: int = Query(..., ge=0, le=120),
    gender: str = Query(..., pattern="^(male|female)$"),
    activity_level: str = Query(..., pattern="^(sedentary|light|moderate|active|very_active)$")
) -> Dict[str, Any]:
    """
    Get personalized nutrient requirements based on demographics
    
    Calculates RDA/DRI values based on age, gender, and activity level
    """
    try:
        # Base nutrient requirements (simplified RDA values)
        base_requirements = {
            "calories": _calculate_calorie_needs(age, gender, activity_level),
            "protein_g": 0.8 * 70,  # 0.8g per kg body weight (assuming 70kg)
            "carbs_g": 130,  # Minimum RDA
            "fat_g": 44,  # Based on 20-35% of calories
            "fiber_g": 25 if gender == "female" else 38,
            "vitamins": {
                "vitamin_c_mg": 75 if gender == "female" else 90,
                "vitamin_d_iu": 600 if age < 70 else 800,
                "vitamin_b12_mcg": 2.4,
                "folate_mcg": 400,
                "vitamin_a_rae": 700 if gender == "female" else 900
            },
            "minerals": {
                "calcium_mg": 1000 if age < 50 else 1200,
                "iron_mg": 18 if gender == "female" and age < 50 else 8,
                "magnesium_mg": 310 if gender == "female" else 400,
                "zinc_mg": 8 if gender == "female" else 11,
                "potassium_mg": 2600 if gender == "female" else 3400
            }
        }
        
        # Adjust for activity level
        activity_multipliers = {
            "sedentary": 1.0,
            "light": 1.1,
            "moderate": 1.2,
            "active": 1.3,
            "very_active": 1.4
        }
        
        multiplier = activity_multipliers[activity_level]
        base_requirements["calories"] = int(base_requirements["calories"] * multiplier)
        base_requirements["protein_g"] = base_requirements["protein_g"] * multiplier
        
        return {
            "user_id": user_id,
            "requirements": base_requirements,
            "notes": [
                "Requirements are estimates based on general population data",
                "Individual needs may vary based on health conditions",
                "Consult healthcare provider for personalized advice"
            ],
            "calculation_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error calculating nutrient requirements: {e}")
        raise HTTPException(status_code=500, detail=f"Error calculating requirements: {str(e)}")

@router.post("/supplements/recommend")
async def recommend_supplements(
    user_id: str,
    health_data: Dict[str, Any],
    genomic_data: Optional[Dict[str, Any]] = None,
    dietary_intake: Optional[Dict[str, float]] = None
) -> List[SupplementRecommendation]:
    """
    Generate personalized supplement recommendations
    
    Analyzes health data, genetic factors, and dietary intake
    to recommend appropriate supplements.
    """
    try:
        recommendations = []
        
        # Vitamin D recommendation
        if genomic_data and genomic_data.get("vitamin_d_synthesis", 1.0) < 0.5:
            recommendations.append(SupplementRecommendation(
                supplement_name="Vitamin D3",
                dosage="2000 IU daily",
                timing="With fat-containing meal",
                reason="Low genetic synthesis capacity",
                confidence_score=0.85,
                interactions=["May increase calcium absorption"],
                cost_estimate=15.0
            ))
        
        # B12 recommendation for vegetarians/vegans
        dietary_restrictions = health_data.get("dietary_restrictions", [])
        if "vegetarian" in dietary_restrictions or "vegan" in dietary_restrictions:
            recommendations.append(SupplementRecommendation(
                supplement_name="Vitamin B12",
                dosage="250 mcg daily",
                timing="Morning with breakfast",
                reason="Limited dietary sources in plant-based diet",
                confidence_score=0.90,
                interactions=[],
                cost_estimate=12.0
            ))
        
        # Omega-3 recommendation
        if not dietary_intake or dietary_intake.get("omega3_g", 0) < 1.0:
            recommendations.append(SupplementRecommendation(
                supplement_name="Omega-3 EPA/DHA",
                dosage="1000 mg daily",
                timing="With meals",
                reason="Insufficient dietary omega-3 intake",
                confidence_score=0.75,
                interactions=["May increase bleeding risk with anticoagulants"],
                cost_estimate=25.0
            ))
        
        # Iron recommendation for women
        if health_data.get("gender") == "female" and health_data.get("age", 0) < 50:
            if health_data.get("hemoglobin", 12.0) < 12.0:
                recommendations.append(SupplementRecommendation(
                    supplement_name="Iron Bisglycinate",
                    dosage="18 mg daily",
                    timing="On empty stomach with vitamin C",
                    reason="Low hemoglobin levels",
                    confidence_score=0.80,
                    interactions=["Avoid with calcium, coffee, tea"],
                    cost_estimate=18.0
                ))
        
        # Magnesium recommendation
        if health_data.get("stress_level", 5) > 7:
            recommendations.append(SupplementRecommendation(
                supplement_name="Magnesium Glycinate",
                dosage="400 mg daily",
                timing="Evening before bed",
                reason="High stress levels and potential deficiency",
                confidence_score=0.70,
                interactions=["May enhance muscle relaxation"],
                cost_estimate=20.0
            ))
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error recommending supplements: {e}")
        raise HTTPException(status_code=500, detail=f"Error recommending supplements: {str(e)}")

@router.get("/analysis/nutrient-gaps/{user_id}")
async def analyze_nutrient_gaps(
    user_id: str,
    dietary_log: List[FoodItem]
) -> Dict[str, Any]:
    """
    Analyze nutrient gaps in current diet
    
    Compares current intake with recommended values
    and identifies deficiencies or excesses.
    """
    try:
        # Calculate current intake
        current_intake = {
            "calories": sum(food.calories_per_100g for food in dietary_log),
            "protein_g": sum(food.protein_g for food in dietary_log),
            "carbs_g": sum(food.carbs_g for food in dietary_log),
            "fat_g": sum(food.fat_g for food in dietary_log),
            "fiber_g": sum(food.fiber_g for food in dietary_log),
            "sodium_mg": sum(food.sodium_mg for food in dietary_log)
        }
        
        # Get recommended intake (placeholder - would use actual user requirements)
        recommended_intake = {
            "calories": 2000,
            "protein_g": 56,
            "carbs_g": 130,
            "fat_g": 44,
            "fiber_g": 25,
            "sodium_mg": 2300
        }
        
        # Calculate gaps
        nutrient_gaps = {}
        recommendations = []
        
        for nutrient, current in current_intake.items():
            recommended = recommended_intake[nutrient]
            gap_percent = ((current - recommended) / recommended) * 100
            
            nutrient_gaps[nutrient] = {
                "current": current,
                "recommended": recommended,
                "gap_percent": gap_percent,
                "status": "adequate" if -10 <= gap_percent <= 10 else 
                         "excess" if gap_percent > 10 else "deficient"
            }
            
            # Generate recommendations
            if gap_percent < -20:
                recommendations.append(f"Increase {nutrient.replace('_', ' ')} intake")
            elif gap_percent > 20:
                recommendations.append(f"Reduce {nutrient.replace('_', ' ')} intake")
        
        return {
            "user_id": user_id,
            "analysis_date": datetime.now().isoformat(),
            "nutrient_gaps": nutrient_gaps,
            "overall_score": _calculate_diet_quality_score(nutrient_gaps),
            "recommendations": recommendations,
            "foods_analyzed": len(dietary_log)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing nutrient gaps: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing gaps: {str(e)}")

@router.get("/foods/categories")
async def get_food_categories() -> Dict[str, List[str]]:
    """Get available food categories and examples"""
    return {
        "protein": ["chicken", "fish", "eggs", "legumes", "tofu"],
        "vegetables": ["spinach", "broccoli", "carrots", "bell peppers"],
        "fruits": ["apples", "berries", "citrus", "bananas"],
        "grains": ["quinoa", "brown rice", "oats", "whole wheat"],
        "dairy": ["yogurt", "milk", "cheese"],
        "fats": ["olive oil", "avocado", "nuts", "seeds"],
        "beverages": ["water", "tea", "coffee", "smoothies"]
    }

# Helper functions
async def _get_filtered_foods(nutrient_profile: NutrientProfile) -> List[FoodItem]:
    """Get foods filtered by dietary restrictions and allergies"""
    # This would query the full food database with filters
    # Placeholder implementation
    all_foods = await search_foods("", None, 100)
    
    filtered_foods = []
    for food in all_foods:
        # Filter by restrictions
        if "vegetarian" in nutrient_profile.dietary_restrictions:
            if food.category.lower() in ["meat", "poultry", "fish"]:
                continue
        
        # Filter by allergies
        skip_food = False
        for allergy in nutrient_profile.allergies:
            if allergy.lower() in food.name.lower():
                skip_food = True
                break
        
        if not skip_food:
            filtered_foods.append(food)
    
    return filtered_foods

def _generate_meal(foods: List[FoodItem], meal_type: str, profile: NutrientProfile) -> List[FoodItem]:
    """Generate a meal based on meal type and profile"""
    # Simplified meal generation
    meal_foods = []
    
    if meal_type == "breakfast":
        # Look for breakfast-appropriate foods
        for food in foods:
            if any(keyword in food.name.lower() for keyword in ["egg", "oat", "yogurt", "fruit"]):
                meal_foods.append(food)
                if len(meal_foods) >= 2:
                    break
    
    elif meal_type == "lunch":
        # Look for lunch foods
        for food in foods:
            if food.category.lower() in ["protein", "vegetables", "grains"]:
                meal_foods.append(food)
                if len(meal_foods) >= 3:
                    break
    
    elif meal_type == "dinner":
        # Similar to lunch but potentially larger portions
        for food in foods:
            if food.category.lower() in ["protein", "vegetables"]:
                meal_foods.append(food)
                if len(meal_foods) >= 3:
                    break
    
    else:  # snack
        for food in foods:
            if any(keyword in food.name.lower() for keyword in ["nut", "fruit", "yogurt"]):
                meal_foods.append(food)
                if len(meal_foods) >= 1:
                    break
    
    return meal_foods

def _calculate_calorie_needs(age: int, gender: str, activity_level: str) -> int:
    """Calculate calorie needs using Mifflin-St Jeor equation (simplified)"""
    # Assuming average height and weight
    if gender == "male":
        bmr = 88.362 + (13.397 * 70) + (4.799 * 175) - (5.677 * age)
    else:
        bmr = 447.593 + (9.247 * 60) + (3.098 * 165) - (4.330 * age)
    
    activity_factors = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9
    }
    
    return int(bmr * activity_factors[activity_level])

def _calculate_diet_quality_score(nutrient_gaps: Dict[str, Any]) -> float:
    """Calculate overall diet quality score"""
    adequate_count = sum(1 for gap in nutrient_gaps.values() if gap["status"] == "adequate")
    total_nutrients = len(nutrient_gaps)
    return (adequate_count / total_nutrients) * 100
