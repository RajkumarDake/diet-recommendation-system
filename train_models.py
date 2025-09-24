#!/usr/bin/env python3
"""
AI Model Training Script for Diet Recommendation System
Trains LSTM, Transformer, and Fusion models with synthetic data
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from models.ai_models import ModelManager
from core.config import settings
from utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

class DataGenerator:
    """Generate synthetic training data for the AI models"""
    
    def __init__(self, n_users: int = 1000):
        self.n_users = n_users
        self.random_state = 42
        np.random.seed(self.random_state)
        random.seed(self.random_state)
    
    def generate_health_data(self) -> pd.DataFrame:
        """Generate synthetic health time-series data"""
        logger.info(f"Generating health data for {self.n_users} users...")
        
        data = []
        
        for user_id in range(self.n_users):
            # Generate 30-90 days of data per user
            n_days = np.random.randint(30, 91)
            base_date = datetime.now() - timedelta(days=n_days)
            
            # User characteristics
            age = np.random.randint(18, 80)
            gender = np.random.choice(['male', 'female'])
            base_weight = np.random.normal(70, 15) if gender == 'male' else np.random.normal(60, 12)
            base_height = np.random.normal(175, 8) if gender == 'male' else np.random.normal(165, 7)
            
            # Generate time series with trends
            weight_trend = np.random.normal(0, 0.1, n_days).cumsum()
            bp_trend = np.random.normal(0, 0.5, n_days).cumsum()
            
            for day in range(n_days):
                current_date = base_date + timedelta(days=day)
                
                # Health metrics with realistic variations
                weight = max(40, base_weight + weight_trend[day] + np.random.normal(0, 0.5))
                height = base_height + np.random.normal(0, 0.1)
                bmi = weight / ((height/100) ** 2)
                
                # Blood pressure with age correlation
                bp_systolic = 90 + (age * 0.5) + bp_trend[day] + np.random.normal(0, 10)
                bp_diastolic = 60 + (age * 0.3) + (bp_trend[day] * 0.6) + np.random.normal(0, 5)
                
                # Other metrics
                cholesterol = 150 + (age * 1.2) + np.random.normal(0, 30)
                glucose = 80 + np.random.normal(0, 15) + (0.1 * max(0, bmi - 25))
                hdl = 45 + np.random.normal(0, 10) + (5 if gender == 'female' else 0)
                ldl = cholesterol - hdl - 20
                triglycerides = 100 + np.random.normal(0, 40) + (2 * max(0, bmi - 25))
                
                data.append({
                    'user_id': f'user_{user_id:04d}',
                    'timestamp': current_date,
                    'age': age,
                    'gender': 1 if gender == 'male' else 0,
                    'height': height,
                    'weight': weight,
                    'bmi': bmi,
                    'cholesterol_total': max(100, cholesterol),
                    'cholesterol_hdl': max(20, hdl),
                    'cholesterol_ldl': max(50, ldl),
                    'blood_pressure_systolic': max(80, bp_systolic),
                    'blood_pressure_diastolic': max(50, bp_diastolic),
                    'glucose': max(60, glucose),
                    'triglycerides': max(50, triglycerides)
                })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} health data points")
        return df
    
    def generate_genomic_data(self) -> pd.DataFrame:
        """Generate synthetic genomic SNP data"""
        logger.info(f"Generating genomic data for {self.n_users} users...")
        
        # Common SNPs related to nutrition and metabolism
        snps = [
            'rs1801282',  # PPARG - fat metabolism
            'rs1801133',  # MTHFR - folate metabolism
            'rs662799',   # APOA5 - triglycerides
            'rs5918',     # ITGB3 - platelet function
            'rs1042713',  # ADRB2 - beta-2 adrenergic receptor
            'rs1042714',  # ADRB2 - beta-2 adrenergic receptor
            'rs4680',     # COMT - dopamine metabolism
            'rs1799971',  # OPRM1 - opioid receptor
            'rs16969968', # CHRNA5 - nicotinic receptor
            'rs1051730'   # CHRNA3 - nicotinic receptor
        ]
        
        data = []
        for user_id in range(self.n_users):
            user_data = {'user_id': f'user_{user_id:04d}'}
            
            for snp in snps:
                # Generate realistic genotype frequencies
                genotypes = ['0/0', '0/1', '1/1', './.']
                weights = [0.5, 0.35, 0.12, 0.03]  # Realistic population frequencies
                genotype = np.random.choice(genotypes, p=weights)
                user_data[snp] = genotype
            
            # Add ancestry information
            ancestries = ['European', 'Asian', 'African', 'Hispanic', 'Mixed']
            user_data['ancestry'] = np.random.choice(ancestries, p=[0.4, 0.2, 0.15, 0.15, 0.1])
            
            data.append(user_data)
        
        df = pd.DataFrame(data)
        logger.info(f"Generated genomic data for {len(df)} users")
        return df
    
    def generate_mental_health_data(self) -> pd.DataFrame:
        """Generate synthetic mental health data"""
        logger.info(f"Generating mental health data for {self.n_users} users...")
        
        data = []
        for user_id in range(self.n_users):
            # Generate correlated mental health metrics
            base_stress = np.random.randint(1, 11)
            
            # Correlated metrics (stress affects other factors)
            anxiety = max(1, min(10, base_stress + np.random.randint(-2, 3)))
            depression = max(1, min(10, int(base_stress * 0.7) + np.random.randint(-2, 3)))
            sleep_quality = max(1, min(10, 11 - int(base_stress * 0.8) + np.random.randint(-1, 2)))
            mood = max(1, min(10, 11 - int(base_stress * 0.6) + np.random.randint(-2, 3)))
            cognitive = max(1, min(10, 11 - int(base_stress * 0.5) + np.random.randint(-1, 2)))
            social_support = np.random.randint(3, 11)
            
            data.append({
                'user_id': f'user_{user_id:04d}',
                'stress_level': base_stress,
                'anxiety_score': anxiety,
                'depression_score': depression,
                'sleep_quality': sleep_quality,
                'mood_rating': mood,
                'cognitive_function': cognitive,
                'social_support': social_support
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated mental health data for {len(df)} users")
        return df
    
    def generate_nutrition_labels(self, health_df: pd.DataFrame, genomic_df: pd.DataFrame, 
                                mental_health_df: pd.DataFrame) -> pd.DataFrame:
        """Generate target nutrition labels based on input data"""
        logger.info("Generating nutrition labels...")
        
        # Get unique users
        users = health_df['user_id'].unique()
        labels = []
        
        for user_id in users:
            user_health = health_df[health_df['user_id'] == user_id].iloc[-1]  # Latest data
            user_genomic = genomic_df[genomic_df['user_id'] == user_id].iloc[0]
            user_mental = mental_health_df[mental_health_df['user_id'] == user_id].iloc[0]
            
            # Determine nutrition category based on health metrics
            bmi = user_health['bmi']
            cholesterol = user_health['cholesterol_total']
            bp_systolic = user_health['blood_pressure_systolic']
            stress = user_mental['stress_level']
            
            # Rule-based label generation (simplified)
            if bmi > 30:
                nutrition_category = 0  # Low Carb, High Protein
            elif cholesterol > 240 or bp_systolic > 140:
                nutrition_category = 1  # Mediterranean Diet
            elif stress > 7:
                nutrition_category = 6  # Anti-Inflammatory Diet
            elif bmi < 18.5:
                nutrition_category = 5  # Balanced Macronutrients
            else:
                nutrition_category = np.random.choice([1, 2, 5], p=[0.4, 0.3, 0.3])
            
            # Health risk level
            risk_factors = 0
            if bmi > 30: risk_factors += 2
            if cholesterol > 240: risk_factors += 2
            if bp_systolic > 140: risk_factors += 2
            if user_health['glucose'] > 126: risk_factors += 2
            if stress > 8: risk_factors += 1
            
            health_risk = min(4, risk_factors)
            
            # Metabolic rate (simplified calculation)
            age = user_health['age']
            weight = user_health['weight']
            height = user_health['height']
            gender = user_health['gender']
            
            # Harris-Benedict equation
            if gender == 1:  # male
                bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
            else:  # female
                bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
            
            # Activity multiplier (random for now)
            activity_multiplier = np.random.uniform(1.2, 1.9)
            metabolic_rate = bmr * activity_multiplier
            
            # Caloric needs
            caloric_needs = metabolic_rate + np.random.normal(0, 100)
            
            labels.append({
                'user_id': user_id,
                'nutrition_category': nutrition_category,
                'health_risk': health_risk,
                'metabolic_rate': metabolic_rate,
                'caloric_needs': caloric_needs
            })
        
        df = pd.DataFrame(labels)
        logger.info(f"Generated labels for {len(df)} users")
        return df

async def train_models():
    """Main training function"""
    logger.info("Starting AI Model Training for Diet Recommendation System")
    
    try:
        # Generate training data
        logger.info("Generating synthetic training data...")
        data_generator = DataGenerator(n_users=2000)  # Increased for better training
        
        health_data = data_generator.generate_health_data()
        genomic_data = data_generator.generate_genomic_data()
        mental_health_data = data_generator.generate_mental_health_data()
        nutrition_labels = data_generator.generate_nutrition_labels(
            health_data, genomic_data, mental_health_data
        )
        
        # Save datasets
        datasets_dir = Path("./datasets")
        datasets_dir.mkdir(exist_ok=True)
        
        health_data.to_csv(datasets_dir / "health_data.csv", index=False)
        genomic_data.to_csv(datasets_dir / "genomic_data.csv", index=False)
        mental_health_data.to_csv(datasets_dir / "mental_health_data.csv", index=False)
        nutrition_labels.to_csv(datasets_dir / "nutrition_labels.csv", index=False)
        
        logger.info("Training datasets saved successfully")
        
        # Initialize model manager
        logger.info("Initializing AI Model Manager...")
        model_manager = ModelManager()
        await model_manager.load_models()
        
        # Train models
        logger.info("Starting model training...")
        training_results = await model_manager.train_models(
            health_data=health_data,
            genomic_data=genomic_data,
            mental_health_data=mental_health_data,
            nutrition_labels=nutrition_labels
        )
        
        logger.info("Model training completed successfully!")
        logger.info(f"Training Results: {training_results}")
        
        # Test model status
        status = await model_manager.get_model_status()
        logger.info(f"Final Model Status: {status}")
        
        return training_results
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

if __name__ == "__main__":
    print("AI-Powered Diet Recommendation System - Model Training")
    print("=" * 60)
    print("Training Configuration:")
    print(f"- LSTM Epochs: 50")
    print(f"- Transformer Epochs: 50") 
    print(f"- Fusion Epochs: 50")
    print(f"- Batch Size: 32")
    print(f"- Users: 2000")
    print("=" * 60)
    
    # Run training
    try:
        results = asyncio.run(train_models())
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("All AI models are now trained and ready:")
        print("✅ LSTM Health Model - Trained")
        print("✅ Transformer Genomic Model - Trained") 
        print("✅ Fusion Model - Trained")
        print("="*60)
        print("\nYour system is ready for real AI-powered nutrition recommendations!")
        print("Start the backend and frontend to test the form.")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print("Models may be partially trained. Please run training again.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("Please check the error above and try again.")
        sys.exit(1)
