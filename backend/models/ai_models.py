"""
AI Models Manager
Centralized management of LSTM, Transformer, and Fusion models
"""

import asyncio
import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from pathlib import Path

from .lstm_health_model import LSTMHealthModel
from .transformer_genomic_model import TransformerGenomicModel
from .fusion_model import FusionModel
from core.config import settings

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Centralized manager for all AI models in the nutrition system
    """
    
    def __init__(self):
        """Initialize model manager"""
        self.lstm_model: Optional[LSTMHealthModel] = None
        self.transformer_model: Optional[TransformerGenomicModel] = None
        self.fusion_model: Optional[FusionModel] = None
        
        self.models_loaded = False
        self.training_in_progress = False
        self.demo_mode = True  # Enable demo mode for testing
        
    async def load_models(self):
        """Load all pre-trained models"""
        try:
            logger.info("Loading AI models...")
            
            # Initialize models
            self.lstm_model = LSTMHealthModel(
                sequence_length=30,
                n_features=len(settings.HEALTH_FEATURES),
                lstm_units=[128, 64, 32],
                dropout_rate=0.2,
                learning_rate=settings.LEARNING_RATE
            )
            
            self.transformer_model = TransformerGenomicModel(
                max_snps=1000,
                d_model=256,
                num_heads=8,
                num_layers=6,
                dff=512,
                dropout_rate=0.1,
                learning_rate=settings.LEARNING_RATE
            )
            
            # Try to load pre-trained models
            try:
                self.lstm_model.load_model(settings.LSTM_MODEL_PATH.replace('.h5', ''))
                logger.info("LSTM model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load LSTM model: {e}. Will need training.")
            
            try:
                self.transformer_model.load_model(settings.TRANSFORMER_MODEL_PATH)
                logger.info("Transformer model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load Transformer model: {e}. Will need training.")
            
            # Initialize fusion model if base models are available
            if self.lstm_model.is_trained and self.transformer_model.is_trained:
                self.fusion_model = FusionModel(
                    lstm_model=self.lstm_model,
                    transformer_model=self.transformer_model,
                    fusion_dim=256,
                    num_attention_heads=8,
                    dropout_rate=0.2,
                    learning_rate=settings.LEARNING_RATE
                )
                
                try:
                    self.fusion_model.load_model(settings.FUSION_MODEL_PATH.replace('.h5', ''))
                    logger.info("Fusion model loaded successfully")
                except Exception as e:
                    logger.warning(f"Could not load Fusion model: {e}. Will need training.")
            
            self.models_loaded = True
            logger.info("Model loading completed")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    async def train_models(self, 
                          health_data: pd.DataFrame,
                          genomic_data: pd.DataFrame,
                          mental_health_data: pd.DataFrame,
                          nutrition_labels: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Train all models sequentially
        
        Args:
            health_data: Health time-series data
            genomic_data: Genomic SNP data
            mental_health_data: Mental health metrics
            
        Returns:
            Training results dictionary
        """
        if self.training_in_progress:
            raise ValueError("Training already in progress")
        
        self.training_in_progress = True
        results = {}
        
        try:
            logger.info("Starting comprehensive model training...")
            
            # 1. Train LSTM model
            logger.info("Training LSTM health model...")
            
            # Prepare training data for LSTM
            if nutrition_labels is not None:
                # Get latest health data per user and merge with labels
                latest_health = health_data.groupby('user_id').last().reset_index()
                # Remove timestamp for single-point training
                if 'timestamp' in latest_health.columns:
                    latest_health = latest_health.drop('timestamp', axis=1)
                training_data = latest_health.merge(nutrition_labels, on='user_id', how='inner')
                logger.info(f"Prepared training data with {len(training_data)} samples")
            else:
                training_data = health_data
            
            target_cols = ['nutrition_category', 'health_risk', 'metabolic_rate', 'caloric_needs']
            lstm_history = self.lstm_model.train(
                data=training_data,
                target_cols=target_cols,
                epochs=settings.LSTM_EPOCHS,
                batch_size=settings.BATCH_SIZE
            )
            results['lstm_training'] = lstm_history
            
            # Save LSTM model
            self.lstm_model.save_model(settings.LSTM_MODEL_PATH.replace('.h5', ''))
            
            # 2. Train Transformer model
            logger.info("Training Transformer genomic model...")
            
            # Prepare genomic targets (placeholder - would come from real data)
            n_samples = len(genomic_data)
            genomic_targets = {
                'metabolism_efficiency': np.random.randint(0, 6, n_samples),
                'absorption_capacity': np.random.randint(0, 5, n_samples),
                'sensitivity_risk': np.random.randint(0, 4, n_samples),
                'inflammation_response': np.random.randint(0, 3, n_samples),
                'vitamin_d_synthesis': np.random.rand(n_samples),
                'caffeine_metabolism': np.random.rand(n_samples)
            }
            
            transformer_history = self.transformer_model.train(
                genomic_data=genomic_data,
                target_data=genomic_targets,
                epochs=settings.TRANSFORMER_EPOCHS,
                batch_size=settings.BATCH_SIZE
            )
            results['transformer_training'] = transformer_history
            
            # Save Transformer model
            self.transformer_model.save_model(settings.TRANSFORMER_MODEL_PATH)
            
            # 3. Train Fusion model
            logger.info("Training Fusion model...")
            self.fusion_model = FusionModel(
                lstm_model=self.lstm_model,
                transformer_model=self.transformer_model,
                fusion_dim=256,
                num_attention_heads=8,
                dropout_rate=0.2,
                learning_rate=settings.LEARNING_RATE
            )
            
            fusion_history = self.fusion_model.train(
                health_data=health_data,
                genomic_data=genomic_data,
                mental_health_data=mental_health_data,
                epochs=settings.FUSION_EPOCHS,
                batch_size=settings.BATCH_SIZE
            )
            results['fusion_training'] = fusion_history
            
            # Save Fusion model
            self.fusion_model.save_model(settings.FUSION_MODEL_PATH.replace('.h5', ''))
            
            logger.info("All models trained successfully!")
            results['status'] = 'success'
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            results['status'] = 'error'
            results['error'] = str(e)
            raise
        
        finally:
            self.training_in_progress = False
        
        return results
    
    async def get_health_predictions(self, health_sequence: np.ndarray) -> Dict[str, Any]:
        """
        Get predictions from LSTM health model
        
        Args:
            health_sequence: Health time-series data
            
        Returns:
            Health predictions
        """
        if not self.lstm_model or not self.lstm_model.is_trained:
            raise ValueError("LSTM model not available or not trained")
        
        return self.lstm_model.predict(health_sequence)
    
    async def get_genomic_predictions(self, genomic_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get predictions from Transformer genomic model
        
        Args:
            genomic_data: Genomic SNP data
            
        Returns:
            Genomic predictions
        """
        if not self.transformer_model or not self.transformer_model.is_trained:
            raise ValueError("Transformer model not available or not trained")
        
        return self.transformer_model.predict(genomic_data)
    
    async def get_comprehensive_recommendations(self,
                                             health_sequence: np.ndarray,
                                             genomic_data: pd.DataFrame,
                                             mental_health_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Get comprehensive recommendations from fusion model
        
        Args:
            health_sequence: Health time-series data
            genomic_data: Genomic SNP data
            mental_health_data: Mental health metrics
            
        Returns:
            Comprehensive nutrition recommendations
        """
        if self.demo_mode:
            # Return demo recommendations for testing
            return self._generate_demo_recommendations(health_sequence, genomic_data, mental_health_data)
        
        if not self.fusion_model or not self.fusion_model.is_trained:
            raise ValueError("Fusion model not available or not trained")
        
        return self.fusion_model.predict(health_sequence, genomic_data, mental_health_data)
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        return {
            'models_loaded': self.models_loaded,
            'training_in_progress': self.training_in_progress,
            'demo_mode': self.demo_mode,
            'lstm_model': {
                'available': self.lstm_model is not None,
                'trained': self.demo_mode or (self.lstm_model.is_trained if self.lstm_model else False)
            },
            'transformer_model': {
                'available': self.transformer_model is not None,
                'trained': self.demo_mode or (self.transformer_model.is_trained if self.transformer_model else False)
            },
            'fusion_model': {
                'available': self.fusion_model is not None,
                'trained': self.demo_mode or (self.fusion_model.is_trained if self.fusion_model else False)
            }
        }
    
    async def retrain_model(self, 
                           model_type: str,
                           training_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Retrain a specific model
        
        Args:
            model_type: Type of model to retrain ('lstm', 'transformer', 'fusion')
            training_data: Training data dictionary
            
        Returns:
            Retraining results
        """
        if self.training_in_progress:
            raise ValueError("Training already in progress")
        
        self.training_in_progress = True
        
        try:
            if model_type == 'lstm':
                if not self.lstm_model:
                    raise ValueError("LSTM model not initialized")
                
                history = self.lstm_model.train(
                    data=training_data['health_data'],
                    target_cols=['nutrition_category', 'health_risk', 'metabolic_rate', 'caloric_needs'],
                    epochs=settings.LSTM_EPOCHS,
                    batch_size=settings.BATCH_SIZE
                )
                
                self.lstm_model.save_model(settings.LSTM_MODEL_PATH.replace('.h5', ''))
                return {'status': 'success', 'history': history}
                
            elif model_type == 'transformer':
                if not self.transformer_model:
                    raise ValueError("Transformer model not initialized")
                
                # Prepare targets (placeholder)
                n_samples = len(training_data['genomic_data'])
                targets = {
                    'metabolism_efficiency': np.random.randint(0, 6, n_samples),
                    'absorption_capacity': np.random.randint(0, 5, n_samples),
                    'sensitivity_risk': np.random.randint(0, 4, n_samples),
                    'inflammation_response': np.random.randint(0, 3, n_samples),
                    'vitamin_d_synthesis': np.random.rand(n_samples),
                    'caffeine_metabolism': np.random.rand(n_samples)
                }
                
                history = self.transformer_model.train(
                    genomic_data=training_data['genomic_data'],
                    target_data=targets,
                    epochs=settings.TRANSFORMER_EPOCHS,
                    batch_size=settings.BATCH_SIZE
                )
                
                self.transformer_model.save_model(settings.TRANSFORMER_MODEL_PATH)
                return {'status': 'success', 'history': history}
                
            elif model_type == 'fusion':
                if not self.fusion_model:
                    raise ValueError("Fusion model not initialized")
                
                history = self.fusion_model.train(
                    health_data=training_data['health_data'],
                    genomic_data=training_data['genomic_data'],
                    mental_health_data=training_data['mental_health_data'],
                    epochs=settings.FUSION_EPOCHS,
                    batch_size=settings.BATCH_SIZE
                )
                
                self.fusion_model.save_model(settings.FUSION_MODEL_PATH.replace('.h5', ''))
                return {'status': 'success', 'history': history}
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Error retraining {model_type} model: {e}")
            return {'status': 'error', 'error': str(e)}
        
        finally:
            self.training_in_progress = False
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up model manager resources...")
        
        # Clear models from memory
        self.lstm_model = None
        self.transformer_model = None
        self.fusion_model = None
        
        self.models_loaded = False
        self.training_in_progress = False
        
        logger.info("Model manager cleanup completed")
    
    def _generate_demo_recommendations(self, health_sequence: np.ndarray, 
                                      genomic_data: pd.DataFrame,
                                      mental_health_data: Dict[str, float]) -> Dict[str, Any]:
        """Generate demo recommendations for testing"""
        import random
        
        # Calculate BMI from health sequence
        bmi = health_sequence[-1][2] if len(health_sequence.shape) > 1 else 25.0
        
        # Determine nutrition category based on BMI and mental health
        if bmi > 30:
            category = "Low Carb, High Protein"
            description = "Focus on lean proteins and minimal carbohydrates for weight management"
        elif bmi < 18.5:
            category = "Balanced Macronutrients"
            description = "Balanced nutrition with emphasis on healthy weight gain"
        else:
            category = "Mediterranean Diet"
            description = "Emphasize olive oil, fish, vegetables, and whole grains for optimal health"
        
        # Generate recommendations based on mental health scores
        stress_level = mental_health_data.get('stress_level', 5)
        sleep_quality = mental_health_data.get('sleep_quality', 5)
        
        supplements = []
        if stress_level > 7:
            supplements.append({
                'supplement': 'Magnesium',
                'confidence': 0.85,
                'reason': 'High stress levels detected - magnesium helps with relaxation'
            })
            supplements.append({
                'supplement': 'B-Complex',
                'confidence': 0.78,
                'reason': 'B vitamins support stress management and energy'
            })
        
        if sleep_quality < 5:
            supplements.append({
                'supplement': 'Melatonin',
                'confidence': 0.72,
                'reason': 'Poor sleep quality detected - melatonin may help regulate sleep cycles'
            })
        
        supplements.append({
            'supplement': 'Vitamin D',
            'confidence': 0.88,
            'reason': 'Essential for immune function and mood regulation'
        })
        
        supplements.append({
            'supplement': 'Omega-3',
            'confidence': 0.82,
            'reason': 'Supports heart health and cognitive function'
        })
        
        # Food recommendations
        if category == "Low Carb, High Protein":
            recommended_foods = ['chicken breast', 'salmon', 'eggs', 'spinach', 'broccoli', 
                               'avocado', 'almonds', 'greek yogurt', 'cauliflower', 'turkey']
            foods_to_limit = ['white bread', 'pasta', 'sugary drinks', 'processed snacks', 'candy']
            meal_ideas = ['Grilled chicken with roasted vegetables', 
                         'Salmon salad with olive oil dressing',
                         'Egg omelet with spinach and mushrooms']
        elif category == "Mediterranean Diet":
            recommended_foods = ['olive oil', 'whole grains', 'fish', 'legumes', 'nuts',
                               'vegetables', 'fruits', 'yogurt', 'herbs', 'garlic']
            foods_to_limit = ['red meat', 'processed foods', 'refined sugars', 'butter', 'salt']
            meal_ideas = ['Greek salad with feta and olives',
                         'Grilled fish with quinoa and vegetables',
                         'Hummus with whole grain pita and vegetables']
        else:
            recommended_foods = ['brown rice', 'lean proteins', 'vegetables', 'fruits', 'nuts',
                               'whole wheat bread', 'dairy', 'eggs', 'beans', 'oats']
            foods_to_limit = ['fried foods', 'excessive sugar', 'processed meats', 'trans fats']
            meal_ideas = ['Balanced bowl with protein, grains, and vegetables',
                         'Whole grain sandwich with lean meat and vegetables',
                         'Oatmeal with fruits and nuts']
        
        # Calculate hydration needs
        weight = health_sequence[-1][3] if len(health_sequence.shape) > 1 else 70
        daily_water = (weight * 0.033) + (0.5 if stress_level > 6 else 0)
        
        return {
            'primary_nutrition_plan': {
                'category': category,
                'confidence': random.uniform(0.75, 0.95),
                'description': description
            },
            'meal_timing': {
                'focus': 'breakfast' if sleep_quality < 5 else 'lunch',
                'confidence': random.uniform(0.7, 0.9)
            },
            'supplements': supplements,
            'portion_guidance': {
                'size': 'small' if bmi > 30 else ('large' if bmi < 18.5 else 'medium'),
                'confidence': random.uniform(0.8, 0.95)
            },
            'hydration': {
                'daily_liters': round(daily_water, 1),
                'recommendations': [
                    f'Aim for {round(daily_water, 1)} liters of water daily',
                    'Drink water before, during, and after exercise',
                    'Monitor urine color as a hydration indicator',
                    'Increase intake during hot weather or exercise'
                ]
            },
            'exercise_nutrition': {
                'timing': 'post-workout',
                'confidence': random.uniform(0.75, 0.9)
            },
            'sensitivity_alerts': [
                {
                    'allergen': 'Gluten',
                    'risk_level': 'low',
                    'confidence': random.uniform(0.3, 0.5)
                }
            ] if random.random() > 0.7 else [],
            'food_recommendations': {
                'recommended_foods': recommended_foods,
                'meal_ideas': meal_ideas,
                'foods_to_limit': foods_to_limit
            },
            'personalization_score': random.uniform(0.82, 0.95)
        }
