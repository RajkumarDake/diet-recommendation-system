"""
LSTM Model for Time-Series Health Data Processing
Processes sequential health metrics to predict nutritional needs
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional
import joblib
import logging

logger = logging.getLogger(__name__)

class LSTMHealthModel:
    """
    Advanced LSTM model for processing time-series health data
    Incorporates attention mechanism and multi-output predictions
    """
    
    def __init__(self, 
                 sequence_length: int = 30,
                 n_features: int = 10,
                 lstm_units: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        """
        Initialize LSTM Health Model
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of health features
            lstm_units: List of LSTM layer units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False
        
    def build_model(self) -> Model:
        """Build the LSTM model architecture"""
        
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # LSTM layers with attention
        x = inputs
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            x = LSTM(units, 
                    return_sequences=return_sequences,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.dropout_rate,
                    name=f'lstm_{i+1}')(x)
            x = BatchNormalization(name=f'bn_lstm_{i+1}')(x)
        
        # Dense layers for feature extraction
        x = Dense(64, activation='relu', name='dense_1')(x)
        x = Dropout(self.dropout_rate, name='dropout_1')(x)
        x = Dense(32, activation='relu', name='dense_2')(x)
        x = Dropout(self.dropout_rate, name='dropout_2')(x)
        
        # Multi-output predictions
        # Nutrition category prediction (0-6, so 7 categories)
        nutrition_output = Dense(7, activation='softmax', name='nutrition_category')(x)
        
        # Health risk assessment
        risk_output = Dense(5, activation='softmax', name='health_risk')(x)
        
        # Metabolic rate prediction
        metabolic_output = Dense(1, activation='linear', name='metabolic_rate')(x)
        
        # Caloric needs prediction
        caloric_output = Dense(1, activation='linear', name='caloric_needs')(x)
        
        # Create model
        model = Model(inputs=inputs, 
                     outputs=[nutrition_output, risk_output, metabolic_output, caloric_output])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss={
                'nutrition_category': 'sparse_categorical_crossentropy',
                'health_risk': 'sparse_categorical_crossentropy', 
                'metabolic_rate': 'mse',
                'caloric_needs': 'mse'
            },
            loss_weights={
                'nutrition_category': 1.0,
                'health_risk': 0.8,
                'metabolic_rate': 0.6,
                'caloric_needs': 0.6
            },
            metrics={
                'nutrition_category': ['accuracy'],
                'health_risk': ['accuracy'],
                'metabolic_rate': ['mae'],
                'caloric_needs': ['mae']
            }
        )
        
        return model
    
    def prepare_sequences(self, data: pd.DataFrame, target_cols: List[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Prepare sequential data for LSTM training
        
        Args:
            data: Health data DataFrame
            target_cols: Target column names
            
        Returns:
            Tuple of (X_sequences, y_dict)
        """
        # Check if we have timestamp column for time series
        has_timestamp = 'timestamp' in data.columns
        
        if has_timestamp:
            # Sort by user_id and timestamp
            data = data.sort_values(['user_id', 'timestamp'])
        
        # Scale features
        exclude_cols = ['user_id'] + target_cols
        if has_timestamp:
            exclude_cols.append('timestamp')
        
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        if len(data) == 0:
            raise ValueError("No data available for training")
        
        logger.info(f"Training data shape: {data.shape}")
        logger.info(f"Feature columns: {feature_cols}")
        logger.info(f"Target columns: {target_cols}")
        logger.info(f"Has timestamp: {has_timestamp}")
        
        X_scaled = self.scaler.fit_transform(data[feature_cols])
        
        # Create sequences
        X_sequences = []
        y_dict = {col: [] for col in target_cols}
        
        if has_timestamp:
            # Time series approach
            for user_id in data['user_id'].unique():
                user_data = data[data['user_id'] == user_id]
                user_X = X_scaled[data['user_id'] == user_id]
                
                if len(user_data) >= self.sequence_length:
                    for i in range(len(user_data) - self.sequence_length + 1):
                        # Input sequence
                        X_sequences.append(user_X[i:i + self.sequence_length])
                        
                        # Target values (last timestep)
                        for col in target_cols:
                            y_dict[col].append(user_data.iloc[i + self.sequence_length - 1][col])
        else:
            # Single point approach - create artificial sequences by repeating the data point
            for i, row in data.iterrows():
                # Create sequence by repeating the same data point
                single_point = X_scaled[i:i+1]  # Get single scaled row
                sequence = np.repeat(single_point, self.sequence_length, axis=0)
                X_sequences.append(sequence)
                
                # Target values
                for col in target_cols:
                    y_dict[col].append(row[col])
        
        logger.info(f"Created {len(X_sequences)} sequences")
        
        if len(X_sequences) == 0:
            raise ValueError("No valid sequences could be created from the data")
        
        X_sequences = np.array(X_sequences)
        logger.info(f"Final X_sequences shape: {X_sequences.shape}")
        
        # Process targets
        for col in target_cols:
            y_dict[col] = np.array(y_dict[col])
            
        return X_sequences, y_dict
    
    def train(self, 
              data: pd.DataFrame,
              target_cols: List[str],
              validation_split: float = 0.2,
              epochs: int = 100,
              batch_size: int = 32) -> Dict:
        """
        Train the LSTM model
        
        Args:
            data: Training data
            target_cols: Target columns
            validation_split: Validation split ratio
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        logger.info("Preparing training data...")
        
        # Prepare sequences
        X, y_dict = self.prepare_sequences(data, target_cols)
        
        # Build model
        self.model = self.build_model()
        
        logger.info(f"Model architecture:\n{self.model.summary()}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7)
        ]
        
        # Train model
        logger.info("Starting model training...")
        history = self.model.fit(
            X, y_dict,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        logger.info("Model training completed!")
        
        return history.history
    
    def predict(self, health_sequence: np.ndarray) -> Dict:
        """
        Make predictions for health sequence
        
        Args:
            health_sequence: Health data sequence
            
        Returns:
            Prediction dictionary
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure correct shape
        if len(health_sequence.shape) == 2:
            health_sequence = health_sequence.reshape(1, *health_sequence.shape)
        
        # Scale input
        health_sequence_scaled = self.scaler.transform(
            health_sequence.reshape(-1, health_sequence.shape[-1])
        ).reshape(health_sequence.shape)
        
        # Make predictions
        predictions = self.model.predict(health_sequence_scaled, verbose=0)
        
        nutrition_pred, risk_pred, metabolic_pred, caloric_pred = predictions
        
        return {
            'nutrition_category': {
                'probabilities': nutrition_pred[0].tolist(),
                'predicted_class': int(np.argmax(nutrition_pred[0])),
                'confidence': float(np.max(nutrition_pred[0]))
            },
            'health_risk': {
                'probabilities': risk_pred[0].tolist(),
                'predicted_class': int(np.argmax(risk_pred[0])),
                'confidence': float(np.max(risk_pred[0]))
            },
            'metabolic_rate': float(metabolic_pred[0][0]),
            'caloric_needs': float(caloric_pred[0][0])
        }
    
    def save_model(self, filepath: str):
        """Save the trained model and preprocessors"""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model
        self.model.save(f"{filepath}.h5")
        
        # Save preprocessors
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        joblib.dump(self.label_encoders, f"{filepath}_encoders.pkl")
        
        # Save metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'is_trained': self.is_trained
        }
        joblib.dump(metadata, f"{filepath}_metadata.pkl")
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and preprocessors"""
        try:
            # Load model
            self.model = tf.keras.models.load_model(f"{filepath}.h5")
            
            # Load preprocessors
            self.scaler = joblib.load(f"{filepath}_scaler.pkl")
            self.label_encoders = joblib.load(f"{filepath}_encoders.pkl")
            
            # Load metadata
            metadata = joblib.load(f"{filepath}_metadata.pkl")
            self.sequence_length = metadata['sequence_length']
            self.n_features = metadata['n_features']
            self.lstm_units = metadata['lstm_units']
            self.dropout_rate = metadata['dropout_rate']
            self.learning_rate = metadata['learning_rate']
            self.is_trained = metadata['is_trained']
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
