"""
Fusion Model for Combining LSTM and Transformer Outputs
Advanced fusion architecture for personalized nutrition recommendations
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, Concatenate,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import joblib
import logging

from .lstm_health_model import LSTMHealthModel
from .transformer_genomic_model import TransformerGenomicModel

logger = logging.getLogger(__name__)

class FusionModel:
    """
    Advanced Fusion Model combining LSTM and Transformer predictions
    Implements attention-based fusion for personalized nutrition recommendations
    """
    
    def __init__(self,
                 lstm_model: LSTMHealthModel,
                 transformer_model: TransformerGenomicModel,
                 fusion_dim: int = 256,
                 num_attention_heads: int = 8,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.0001):
        """
        Initialize Fusion Model
        
        Args:
            lstm_model: Trained LSTM health model
            transformer_model: Trained Transformer genomic model
            fusion_dim: Fusion layer dimension
            num_attention_heads: Number of attention heads for fusion
            dropout_rate: Dropout rate
            learning_rate: Learning rate
        """
        self.lstm_model = lstm_model
        self.transformer_model = transformer_model
        self.fusion_dim = fusion_dim
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Nutrition categories
        self.nutrition_categories = {
            0: "Low Carb, High Protein",
            1: "Mediterranean Diet", 
            2: "Low Fat, High Fiber",
            3: "Ketogenic Diet",
            4: "Plant-Based Diet",
            5: "Balanced Macronutrients",
            6: "Anti-Inflammatory Diet",
            7: "Personalized Micronutrient Focus"
        }
        
        # Food recommendations database
        self.food_recommendations = {
            'high_protein': ['lean chicken', 'fish', 'eggs', 'legumes', 'quinoa'],
            'low_carb': ['leafy greens', 'broccoli', 'cauliflower', 'avocado', 'nuts'],
            'anti_inflammatory': ['turmeric', 'ginger', 'berries', 'fatty fish', 'olive oil'],
            'high_fiber': ['oats', 'beans', 'apples', 'chia seeds', 'vegetables'],
            'micronutrient_rich': ['spinach', 'sweet potato', 'salmon', 'almonds', 'yogurt']
        }
    
    def build_fusion_model(self) -> Model:
        """Build the fusion model architecture"""
        
        # Input layers for different data types
        health_input = Input(shape=(30, 10), name='health_sequence')  # LSTM input
        genomic_snp_input = Input(shape=(1000,), name='genomic_snps')  # Transformer input
        genomic_genotype_input = Input(shape=(1000,), name='genomic_genotypes')
        mental_health_input = Input(shape=(7,), name='mental_health')  # Additional features
        
        # Get LSTM predictions (freeze LSTM model)
        lstm_features = self.lstm_model.model(health_input)
        
        # Get Transformer predictions (freeze Transformer model)
        transformer_features = self.transformer_model.model([genomic_snp_input, genomic_genotype_input])
        
        # Process mental health features
        mental_features = Dense(64, activation='relu', name='mental_dense1')(mental_health_input)
        mental_features = Dropout(self.dropout_rate)(mental_features)
        mental_features = Dense(32, activation='relu', name='mental_dense2')(mental_features)
        
        # Extract key features from LSTM outputs
        lstm_nutrition = lstm_features[0]  # nutrition_category
        lstm_risk = lstm_features[1]       # health_risk
        lstm_metabolic = lstm_features[2]  # metabolic_rate
        lstm_caloric = lstm_features[3]    # caloric_needs
        
        # Extract key features from Transformer outputs
        transformer_metabolism = transformer_features[0]  # metabolism_efficiency
        transformer_absorption = transformer_features[1]  # absorption_capacity
        transformer_sensitivity = transformer_features[2] # sensitivity_risk
        transformer_inflammation = transformer_features[3] # inflammation_response
        
        # Create feature vectors for fusion
        lstm_vector = Concatenate(name='lstm_concat')([
            lstm_nutrition, lstm_risk, 
            tf.expand_dims(lstm_metabolic, -1), 
            tf.expand_dims(lstm_caloric, -1)
        ])
        
        transformer_vector = Concatenate(name='transformer_concat')([
            transformer_metabolism, transformer_absorption,
            transformer_sensitivity, transformer_inflammation
        ])
        
        # Project to same dimension for fusion
        lstm_projected = Dense(self.fusion_dim, activation='relu', name='lstm_projection')(lstm_vector)
        transformer_projected = Dense(self.fusion_dim, activation='relu', name='transformer_projection')(transformer_vector)
        mental_projected = Dense(self.fusion_dim, activation='relu', name='mental_projection')(mental_features)
        
        # Stack features for attention
        stacked_features = tf.stack([lstm_projected, transformer_projected, mental_projected], axis=1)
        
        # Multi-head attention fusion
        attention_output = MultiHeadAttention(
            num_heads=self.num_attention_heads,
            key_dim=self.fusion_dim // self.num_attention_heads,
            name='fusion_attention'
        )(stacked_features, stacked_features)
        
        # Layer normalization
        attention_output = LayerNormalization(epsilon=1e-6)(stacked_features + attention_output)
        
        # Global pooling to combine attended features
        fused_features = GlobalAveragePooling1D()(attention_output)
        
        # Additional fusion layers
        x = Dense(512, activation='relu', name='fusion_dense1')(fused_features)
        x = BatchNormalization(name='fusion_bn1')(x)
        x = Dropout(self.dropout_rate)(x)
        
        x = Dense(256, activation='relu', name='fusion_dense2')(x)
        x = BatchNormalization(name='fusion_bn2')(x)
        x = Dropout(self.dropout_rate)(x)
        
        x = Dense(128, activation='relu', name='fusion_dense3')(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Final prediction outputs
        # Primary nutrition recommendation
        nutrition_recommendation = Dense(8, activation='softmax', name='nutrition_recommendation')(x)
        
        # Meal timing optimization
        meal_timing = Dense(4, activation='softmax', name='meal_timing')(x)  # breakfast, lunch, dinner, snack focus
        
        # Supplement recommendations
        supplement_needs = Dense(6, activation='sigmoid', name='supplement_needs')(x)  # multi-label
        
        # Portion size recommendations
        portion_sizes = Dense(3, activation='softmax', name='portion_sizes')(x)  # small, medium, large
        
        # Hydration needs
        hydration_needs = Dense(1, activation='linear', name='hydration_needs')(x)  # liters per day
        
        # Exercise nutrition timing
        exercise_nutrition = Dense(3, activation='softmax', name='exercise_nutrition')(x)  # pre, during, post
        
        # Food sensitivity alerts
        sensitivity_alerts = Dense(5, activation='sigmoid', name='sensitivity_alerts')(x)  # multi-label
        
        # Create fusion model
        fusion_model = Model(
            inputs=[health_input, genomic_snp_input, genomic_genotype_input, mental_health_input],
            outputs=[
                nutrition_recommendation, meal_timing, supplement_needs,
                portion_sizes, hydration_needs, exercise_nutrition, sensitivity_alerts
            ]
        )
        
        # Compile model
        fusion_model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss={
                'nutrition_recommendation': 'sparse_categorical_crossentropy',
                'meal_timing': 'sparse_categorical_crossentropy',
                'supplement_needs': 'binary_crossentropy',
                'portion_sizes': 'sparse_categorical_crossentropy',
                'hydration_needs': 'mse',
                'exercise_nutrition': 'sparse_categorical_crossentropy',
                'sensitivity_alerts': 'binary_crossentropy'
            },
            loss_weights={
                'nutrition_recommendation': 1.0,
                'meal_timing': 0.8,
                'supplement_needs': 0.7,
                'portion_sizes': 0.6,
                'hydration_needs': 0.5,
                'exercise_nutrition': 0.4,
                'sensitivity_alerts': 0.6
            },
            metrics={
                'nutrition_recommendation': ['accuracy', 'top_k_categorical_accuracy'],
                'meal_timing': ['accuracy'],
                'supplement_needs': ['binary_accuracy'],
                'portion_sizes': ['accuracy'],
                'hydration_needs': ['mae'],
                'exercise_nutrition': ['accuracy'],
                'sensitivity_alerts': ['binary_accuracy']
            }
        )
        
        return fusion_model
    
    def prepare_fusion_data(self, 
                           health_data: pd.DataFrame,
                           genomic_data: pd.DataFrame,
                           mental_health_data: pd.DataFrame) -> Tuple[List[np.ndarray], Dict[str, np.ndarray]]:
        """
        Prepare data for fusion model training
        
        Args:
            health_data: Health time-series data
            genomic_data: Genomic SNP data
            mental_health_data: Mental health metrics
            
        Returns:
            Tuple of (input_list, target_dict)
        """
        # Prepare LSTM input
        health_sequences, _ = self.lstm_model.prepare_sequences(
            health_data, ['nutrition_category', 'health_risk', 'metabolic_rate', 'caloric_needs']
        )
        
        # Prepare Transformer input
        snp_sequences, genotype_sequences = self.transformer_model.preprocess_genomic_data(genomic_data)
        
        # Prepare mental health input
        mental_features = mental_health_data[self.lstm_model.MENTAL_HEALTH_FEATURES].values
        mental_features = self.scaler.fit_transform(mental_features)
        
        inputs = [health_sequences, snp_sequences, genotype_sequences, mental_features]
        
        # Prepare targets (these would come from your labeled dataset)
        targets = {
            'nutrition_recommendation': np.random.randint(0, 8, len(health_sequences)),  # Placeholder
            'meal_timing': np.random.randint(0, 4, len(health_sequences)),
            'supplement_needs': np.random.randint(0, 2, (len(health_sequences), 6)),
            'portion_sizes': np.random.randint(0, 3, len(health_sequences)),
            'hydration_needs': np.random.uniform(1.5, 4.0, len(health_sequences)),
            'exercise_nutrition': np.random.randint(0, 3, len(health_sequences)),
            'sensitivity_alerts': np.random.randint(0, 2, (len(health_sequences), 5))
        }
        
        return inputs, targets
    
    def train(self,
              health_data: pd.DataFrame,
              genomic_data: pd.DataFrame,
              mental_health_data: pd.DataFrame,
              validation_split: float = 0.2,
              epochs: int = 75,
              batch_size: int = 16) -> Dict:
        """
        Train the fusion model
        
        Args:
            health_data: Health time-series data
            genomic_data: Genomic data
            mental_health_data: Mental health data
            validation_split: Validation split ratio
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        logger.info("Preparing fusion training data...")
        
        # Prepare data
        inputs, targets = self.prepare_fusion_data(health_data, genomic_data, mental_health_data)
        
        # Freeze base models
        self.lstm_model.model.trainable = False
        self.transformer_model.model.trainable = False
        
        # Build fusion model
        self.model = self.build_fusion_model()
        
        logger.info(f"Fusion model architecture:\n{self.model.summary()}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-8)
        ]
        
        # Train model
        logger.info("Starting fusion model training...")
        history = self.model.fit(
            inputs, targets,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        logger.info("Fusion model training completed!")
        
        return history.history
    
    def predict(self,
                health_sequence: np.ndarray,
                genomic_data: pd.DataFrame,
                mental_health_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate comprehensive nutrition recommendations
        
        Args:
            health_sequence: Health time-series data
            genomic_data: Genomic SNP data
            mental_health_data: Mental health metrics
            
        Returns:
            Comprehensive recommendation dictionary
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Fusion model must be trained before making predictions")
        
        # Prepare inputs
        health_input = health_sequence.reshape(1, *health_sequence.shape)
        
        snp_sequences, genotype_sequences = self.transformer_model.preprocess_genomic_data(genomic_data)
        snp_input = snp_sequences[:1]
        genotype_input = genotype_sequences[:1]
        
        mental_input = np.array([[mental_health_data[feat] for feat in self.lstm_model.MENTAL_HEALTH_FEATURES]])
        mental_input = self.scaler.transform(mental_input)
        
        # Make predictions
        predictions = self.model.predict([health_input, snp_input, genotype_input, mental_input], verbose=0)
        
        (nutrition_pred, meal_timing_pred, supplement_pred, portion_pred,
         hydration_pred, exercise_pred, sensitivity_pred) = predictions
        
        # Process predictions
        nutrition_class = int(np.argmax(nutrition_pred[0]))
        nutrition_confidence = float(np.max(nutrition_pred[0]))
        
        # Generate detailed recommendations
        recommendations = {
            'primary_nutrition_plan': {
                'category': self.nutrition_categories[nutrition_class],
                'confidence': nutrition_confidence,
                'description': self._get_nutrition_description(nutrition_class)
            },
            'meal_timing': {
                'focus': ['breakfast', 'lunch', 'dinner', 'snacks'][np.argmax(meal_timing_pred[0])],
                'confidence': float(np.max(meal_timing_pred[0]))
            },
            'supplements': self._get_supplement_recommendations(supplement_pred[0]),
            'portion_guidance': {
                'size': ['small', 'medium', 'large'][np.argmax(portion_pred[0])],
                'confidence': float(np.max(portion_pred[0]))
            },
            'hydration': {
                'daily_liters': float(hydration_pred[0][0]),
                'recommendations': self._get_hydration_tips(float(hydration_pred[0][0]))
            },
            'exercise_nutrition': {
                'timing': ['pre-workout', 'during-workout', 'post-workout'][np.argmax(exercise_pred[0])],
                'confidence': float(np.max(exercise_pred[0]))
            },
            'sensitivity_alerts': self._get_sensitivity_alerts(sensitivity_pred[0]),
            'food_recommendations': self._generate_food_recommendations(nutrition_class, genomic_data),
            'personalization_score': self._calculate_personalization_score(predictions)
        }
        
        return recommendations
    
    def _get_nutrition_description(self, category: int) -> str:
        """Get detailed description for nutrition category"""
        descriptions = {
            0: "Focus on lean proteins and minimal carbohydrates for optimal metabolic health",
            1: "Emphasize olive oil, fish, vegetables, and whole grains for heart health",
            2: "Prioritize fiber-rich foods and limit saturated fats for digestive health",
            3: "High-fat, very low-carb approach for metabolic flexibility",
            4: "Plant-based nutrition with complete protein combinations",
            5: "Balanced macronutrient distribution for general wellness",
            6: "Foods rich in omega-3s and antioxidants to reduce inflammation",
            7: "Targeted micronutrient optimization based on genetic profile"
        }
        return descriptions.get(category, "Personalized nutrition approach")
    
    def _get_supplement_recommendations(self, supplement_probs: np.ndarray) -> List[Dict]:
        """Generate supplement recommendations"""
        supplements = ['Vitamin D', 'Omega-3', 'B-Complex', 'Magnesium', 'Probiotics', 'Iron']
        recommendations = []
        
        for i, prob in enumerate(supplement_probs):
            if prob > 0.5:
                recommendations.append({
                    'supplement': supplements[i],
                    'confidence': float(prob),
                    'reason': f"Based on your genetic and health profile"
                })
        
        return recommendations
    
    def _get_hydration_tips(self, daily_liters: float) -> List[str]:
        """Generate hydration recommendations"""
        tips = [
            f"Aim for {daily_liters:.1f} liters of water daily",
            "Drink water before, during, and after exercise",
            "Monitor urine color as a hydration indicator"
        ]
        
        if daily_liters > 3.0:
            tips.append("Consider electrolyte supplementation with high water intake")
        
        return tips
    
    def _get_sensitivity_alerts(self, sensitivity_probs: np.ndarray) -> List[Dict]:
        """Generate food sensitivity alerts"""
        sensitivities = ['Gluten', 'Dairy', 'Nuts', 'Shellfish', 'Soy']
        alerts = []
        
        for i, prob in enumerate(sensitivity_probs):
            if prob > 0.6:
                alerts.append({
                    'allergen': sensitivities[i],
                    'risk_level': 'high' if prob > 0.8 else 'moderate',
                    'confidence': float(prob)
                })
        
        return alerts
    
    def _generate_food_recommendations(self, nutrition_category: int, genomic_data: pd.DataFrame) -> Dict:
        """Generate specific food recommendations"""
        base_foods = []
        
        if nutrition_category in [0, 3]:  # Low carb/Keto
            base_foods.extend(self.food_recommendations['low_carb'])
            base_foods.extend(self.food_recommendations['high_protein'])
        elif nutrition_category == 1:  # Mediterranean
            base_foods.extend(['olive oil', 'fish', 'vegetables', 'whole grains'])
        elif nutrition_category == 6:  # Anti-inflammatory
            base_foods.extend(self.food_recommendations['anti_inflammatory'])
        
        return {
            'recommended_foods': base_foods[:10],
            'meal_ideas': self._generate_meal_ideas(nutrition_category),
            'foods_to_limit': self._get_foods_to_limit(nutrition_category)
        }
    
    def _generate_meal_ideas(self, category: int) -> List[str]:
        """Generate meal ideas based on nutrition category"""
        meal_ideas = {
            0: ["Grilled chicken with vegetables", "Salmon with asparagus", "Egg omelet with spinach"],
            1: ["Greek salad with olive oil", "Grilled fish with quinoa", "Vegetable pasta"],
            6: ["Turmeric latte", "Berry smoothie", "Ginger tea with honey"]
        }
        return meal_ideas.get(category, ["Balanced meal with protein, vegetables, and healthy fats"])
    
    def _get_foods_to_limit(self, category: int) -> List[str]:
        """Get foods to limit based on nutrition category"""
        limit_foods = {
            0: ["refined carbs", "sugary drinks", "processed foods"],
            3: ["all grains", "fruits", "starchy vegetables"],
            6: ["processed meats", "trans fats", "excessive sugar"]
        }
        return limit_foods.get(category, ["processed foods", "excessive sugar"])
    
    def _calculate_personalization_score(self, predictions: List[np.ndarray]) -> float:
        """Calculate overall personalization confidence score"""
        confidences = []
        
        # Extract confidence scores from predictions
        for pred in predictions[:4]:  # First 4 are categorical
            if len(pred.shape) > 1:
                confidences.append(np.max(pred[0]))
        
        return float(np.mean(confidences)) if confidences else 0.0
    
    def save_model(self, filepath: str):
        """Save the fusion model and preprocessors"""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save fusion model
        self.model.save(f"{filepath}_fusion.h5")
        
        # Save preprocessors
        joblib.dump(self.scaler, f"{filepath}_fusion_scaler.pkl")
        
        # Save metadata
        metadata = {
            'fusion_dim': self.fusion_dim,
            'num_attention_heads': self.num_attention_heads,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'is_trained': self.is_trained,
            'nutrition_categories': self.nutrition_categories,
            'food_recommendations': self.food_recommendations
        }
        joblib.dump(metadata, f"{filepath}_fusion_metadata.pkl")
        
        logger.info(f"Fusion model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the fusion model and preprocessors"""
        try:
            # Load fusion model
            self.model = tf.keras.models.load_model(f"{filepath}_fusion.h5")
            
            # Load preprocessors
            self.scaler = joblib.load(f"{filepath}_fusion_scaler.pkl")
            
            # Load metadata
            metadata = joblib.load(f"{filepath}_fusion_metadata.pkl")
            for key, value in metadata.items():
                setattr(self, key, value)
            
            logger.info(f"Fusion model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading fusion model: {e}")
            raise
