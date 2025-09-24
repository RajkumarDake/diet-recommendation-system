"""
Transformer Model for Genomic Data Analysis
Processes SNP variations and genomic markers for personalized nutrition
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, 
    MultiHeadAttention, GlobalAveragePooling1D, Embedding
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, List, Tuple, Optional
import joblib
import logging

logger = logging.getLogger(__name__)

class TransformerGenomicModel:
    """
    Advanced Transformer model for genomic data analysis
    Processes SNP variations and genetic markers for nutrition recommendations
    """
    
    def __init__(self,
                 max_snps: int = 1000,
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dff: int = 512,
                 dropout_rate: float = 0.1,
                 learning_rate: float = 0.0001):
        """
        Initialize Transformer Genomic Model
        
        Args:
            max_snps: Maximum number of SNPs to process
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dff: Feed-forward dimension
            dropout_rate: Dropout rate
            learning_rate: Learning rate
        """
        self.max_snps = max_snps
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        self.snp_encoder = LabelEncoder()
        self.genotype_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # SNP categories for nutrition-related genes
        self.nutrition_snps = {
            'metabolism': ['rs1801282', 'rs1801133', 'rs662799', 'rs5918'],
            'absorption': ['rs1042713', 'rs1042714', 'rs4680', 'rs1799971'],
            'sensitivity': ['rs16969968', 'rs1051730', 'rs1800497', 'rs6265'],
            'inflammation': ['rs1800629', 'rs361525', 'rs1143634', 'rs16944']
        }
    
    def positional_encoding(self, position: int, d_model: int) -> tf.Tensor:
        """Create positional encoding for transformer"""
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        
        # Apply sin to even indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # Apply cos to odd indices
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def get_angles(self, pos: np.ndarray, i: np.ndarray, d_model: int) -> np.ndarray:
        """Calculate angles for positional encoding"""
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
    def transformer_block(self, inputs: tf.Tensor, name: str) -> tf.Tensor:
        """Single transformer block"""
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            name=f'{name}_attention'
        )(inputs, inputs)
        
        attention_output = Dropout(self.dropout_rate)(attention_output)
        attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        # Feed forward network
        ffn_output = Dense(self.dff, activation='relu', name=f'{name}_ffn1')(attention_output)
        ffn_output = Dense(self.d_model, name=f'{name}_ffn2')(ffn_output)
        ffn_output = Dropout(self.dropout_rate)(ffn_output)
        ffn_output = LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
        
        return ffn_output
    
    def build_model(self) -> Model:
        """Build the Transformer model architecture"""
        
        # Input layers
        snp_input = Input(shape=(self.max_snps,), name='snp_ids')
        genotype_input = Input(shape=(self.max_snps,), name='genotypes')
        
        # Embedding layers
        snp_embedding = Embedding(
            input_dim=10000,  # Vocabulary size for SNP IDs
            output_dim=self.d_model,
            mask_zero=True,
            name='snp_embedding'
        )(snp_input)
        
        genotype_embedding = Embedding(
            input_dim=4,  # 0, 1, 2, 3 for genotype encoding
            output_dim=self.d_model,
            mask_zero=True,
            name='genotype_embedding'
        )(genotype_input)
        
        # Combine embeddings
        combined_embedding = snp_embedding + genotype_embedding
        
        # Add positional encoding
        pos_encoding = self.positional_encoding(self.max_snps, self.d_model)
        combined_embedding += pos_encoding[:, :self.max_snps, :]
        
        # Apply dropout
        x = Dropout(self.dropout_rate)(combined_embedding)
        
        # Transformer blocks
        for i in range(self.num_layers):
            x = self.transformer_block(x, f'transformer_{i}')
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # Dense layers for different predictions
        x = Dense(512, activation='relu', name='dense_1')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(256, activation='relu', name='dense_2')(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Multi-output predictions
        # Nutrient metabolism efficiency
        metabolism_output = Dense(6, activation='softmax', name='metabolism_efficiency')(x)
        
        # Nutrient absorption capacity
        absorption_output = Dense(5, activation='softmax', name='absorption_capacity')(x)
        
        # Food sensitivity risk
        sensitivity_output = Dense(4, activation='softmax', name='sensitivity_risk')(x)
        
        # Inflammation response
        inflammation_output = Dense(3, activation='softmax', name='inflammation_response')(x)
        
        # Vitamin D synthesis
        vitamin_d_output = Dense(1, activation='sigmoid', name='vitamin_d_synthesis')(x)
        
        # Caffeine metabolism
        caffeine_output = Dense(1, activation='sigmoid', name='caffeine_metabolism')(x)
        
        # Create model
        model = Model(
            inputs=[snp_input, genotype_input],
            outputs=[
                metabolism_output, absorption_output, sensitivity_output,
                inflammation_output, vitamin_d_output, caffeine_output
            ]
        )
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss={
                'metabolism_efficiency': 'sparse_categorical_crossentropy',
                'absorption_capacity': 'sparse_categorical_crossentropy',
                'sensitivity_risk': 'sparse_categorical_crossentropy',
                'inflammation_response': 'sparse_categorical_crossentropy',
                'vitamin_d_synthesis': 'mse',
                'caffeine_metabolism': 'binary_crossentropy'
            },
            loss_weights={
                'metabolism_efficiency': 1.0,
                'absorption_capacity': 0.9,
                'sensitivity_risk': 0.8,
                'inflammation_response': 0.7,
                'vitamin_d_synthesis': 0.6,
                'caffeine_metabolism': 0.5
            },
            metrics={
                'metabolism_efficiency': ['accuracy'],
                'absorption_capacity': ['accuracy'],
                'sensitivity_risk': ['accuracy'],
                'inflammation_response': ['accuracy'],
                'vitamin_d_synthesis': ['accuracy'],
                'caffeine_metabolism': ['accuracy']
            }
        )
        
        return model
    
    def preprocess_genomic_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess genomic data for transformer input
        
        Args:
            data: Genomic data DataFrame with SNP columns
            
        Returns:
            Tuple of (snp_sequences, genotype_sequences)
        """
        # Extract SNP data
        snp_columns = [col for col in data.columns if col.startswith('rs')]
        
        snp_sequences = []
        genotype_sequences = []
        
        for _, row in data.iterrows():
            snp_ids = []
            genotypes = []
            
            for snp_col in snp_columns[:self.max_snps]:
                # Encode SNP ID
                snp_id = hash(snp_col) % 10000  # Simple hash for SNP ID
                snp_ids.append(snp_id)
                
                # Encode genotype (0/0, 0/1, 1/1, missing)
                genotype_val = row[snp_col]
                if pd.isna(genotype_val):
                    genotypes.append(0)  # Missing
                elif genotype_val == '0/0':
                    genotypes.append(1)  # Homozygous reference
                elif genotype_val == '0/1' or genotype_val == '1/0':
                    genotypes.append(2)  # Heterozygous
                elif genotype_val == '1/1':
                    genotypes.append(3)  # Homozygous alternate
                else:
                    genotypes.append(0)  # Unknown
            
            # Pad sequences
            while len(snp_ids) < self.max_snps:
                snp_ids.append(0)
                genotypes.append(0)
            
            snp_sequences.append(snp_ids)
            genotype_sequences.append(genotypes)
        
        return np.array(snp_sequences), np.array(genotype_sequences)
    
    def train(self,
              genomic_data: pd.DataFrame,
              target_data: Dict[str, np.ndarray],
              validation_split: float = 0.2,
              epochs: int = 50,
              batch_size: int = 16) -> Dict:
        """
        Train the Transformer model
        
        Args:
            genomic_data: Genomic data DataFrame
            target_data: Dictionary of target arrays
            validation_split: Validation split ratio
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        logger.info("Preprocessing genomic data...")
        
        # Preprocess input data
        snp_sequences, genotype_sequences = self.preprocess_genomic_data(genomic_data)
        
        # Build model
        self.model = self.build_model()
        
        logger.info(f"Model architecture:\n{self.model.summary()}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-8)
        ]
        
        # Train model
        logger.info("Starting model training...")
        history = self.model.fit(
            [snp_sequences, genotype_sequences],
            target_data,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        logger.info("Model training completed!")
        
        return history.history
    
    def predict(self, genomic_data: pd.DataFrame) -> Dict:
        """
        Make predictions for genomic data
        
        Args:
            genomic_data: Genomic data DataFrame
            
        Returns:
            Prediction dictionary
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess input
        snp_sequences, genotype_sequences = self.preprocess_genomic_data(genomic_data)
        
        # Make predictions
        predictions = self.model.predict([snp_sequences, genotype_sequences], verbose=0)
        
        (metabolism_pred, absorption_pred, sensitivity_pred, 
         inflammation_pred, vitamin_d_pred, caffeine_pred) = predictions
        
        results = []
        for i in range(len(genomic_data)):
            result = {
                'metabolism_efficiency': {
                    'probabilities': metabolism_pred[i].tolist(),
                    'predicted_class': int(np.argmax(metabolism_pred[i])),
                    'confidence': float(np.max(metabolism_pred[i]))
                },
                'absorption_capacity': {
                    'probabilities': absorption_pred[i].tolist(),
                    'predicted_class': int(np.argmax(absorption_pred[i])),
                    'confidence': float(np.max(absorption_pred[i]))
                },
                'sensitivity_risk': {
                    'probabilities': sensitivity_pred[i].tolist(),
                    'predicted_class': int(np.argmax(sensitivity_pred[i])),
                    'confidence': float(np.max(sensitivity_pred[i]))
                },
                'inflammation_response': {
                    'probabilities': inflammation_pred[i].tolist(),
                    'predicted_class': int(np.argmax(inflammation_pred[i])),
                    'confidence': float(np.max(inflammation_pred[i]))
                },
                'vitamin_d_synthesis': float(vitamin_d_pred[i][0]),
                'caffeine_metabolism': float(caffeine_pred[i][0])
            }
            results.append(result)
        
        return results[0] if len(results) == 1 else results
    
    def save_model(self, filepath: str):
        """Save the trained model and preprocessors"""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model
        self.model.save(f"{filepath}.h5")
        
        # Save preprocessors
        joblib.dump(self.snp_encoder, f"{filepath}_snp_encoder.pkl")
        joblib.dump(self.genotype_encoder, f"{filepath}_genotype_encoder.pkl")
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        
        # Save metadata
        metadata = {
            'max_snps': self.max_snps,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'is_trained': self.is_trained,
            'nutrition_snps': self.nutrition_snps
        }
        joblib.dump(metadata, f"{filepath}_metadata.pkl")
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and preprocessors"""
        try:
            # Load model
            self.model = tf.keras.models.load_model(f"{filepath}.h5")
            
            # Load preprocessors
            self.snp_encoder = joblib.load(f"{filepath}_snp_encoder.pkl")
            self.genotype_encoder = joblib.load(f"{filepath}_genotype_encoder.pkl")
            self.scaler = joblib.load(f"{filepath}_scaler.pkl")
            
            # Load metadata
            metadata = joblib.load(f"{filepath}_metadata.pkl")
            for key, value in metadata.items():
                setattr(self, key, value)
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
