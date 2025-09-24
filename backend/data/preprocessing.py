"""
Data Preprocessing Pipeline for AI-Powered Nutrition System
Handles data ingestion, cleaning, and preparation from multiple sources
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import requests
import json
from datetime import datetime, timedelta
import asyncio
import aiohttp
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
import joblib

from core.config import settings

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for multi-modal nutrition data
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.data_sources = {
            'nhanes': NHANESProcessor(),
            'genomic': GenomicProcessor(),
            'usda': USDAProcessor(),
            'drugbank': DrugBankProcessor(),
            'mental_health': MentalHealthProcessor()
        }
    
    async def process_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Process data from all sources and return cleaned datasets
        
        Returns:
            Dictionary of processed DataFrames by source
        """
        logger.info("Starting comprehensive data preprocessing...")
        
        processed_data = {}
        
        # Process each data source
        for source_name, processor in self.data_sources.items():
            try:
                logger.info(f"Processing {source_name} data...")
                processed_data[source_name] = await processor.process()
                logger.info(f"Successfully processed {source_name} data: {len(processed_data[source_name])} records")
            except Exception as e:
                logger.error(f"Error processing {source_name} data: {e}")
                processed_data[source_name] = pd.DataFrame()  # Empty DataFrame as fallback
        
        # Cross-validate and align data
        processed_data = self._align_datasets(processed_data)
        
        # Generate synthetic data if needed for training
        processed_data = await self._generate_synthetic_data(processed_data)
        
        logger.info("Data preprocessing completed successfully")
        return processed_data
    
    def _align_datasets(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Align datasets by common identifiers and time periods"""
        
        # Find common time range
        min_date = None
        max_date = None
        
        for df in data.values():
            if 'timestamp' in df.columns and not df.empty:
                df_min = df['timestamp'].min()
                df_max = df['timestamp'].max()
                
                if min_date is None or df_min > min_date:
                    min_date = df_min
                if max_date is None or df_max < max_date:
                    max_date = df_max
        
        # Filter datasets to common time range
        if min_date and max_date:
            for name, df in data.items():
                if 'timestamp' in df.columns and not df.empty:
                    data[name] = df[(df['timestamp'] >= min_date) & (df['timestamp'] <= max_date)]
        
        return data
    
    async def _generate_synthetic_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate synthetic data for training when real data is insufficient"""
        
        # Generate synthetic health data
        if data['nhanes'].empty or len(data['nhanes']) < 1000:
            logger.info("Generating synthetic health data...")
            data['nhanes'] = self._generate_synthetic_health_data(5000)
        
        # Generate synthetic genomic data
        if data['genomic'].empty or len(data['genomic']) < 1000:
            logger.info("Generating synthetic genomic data...")
            data['genomic'] = self._generate_synthetic_genomic_data(5000)
        
        # Generate synthetic mental health data
        if data['mental_health'].empty or len(data['mental_health']) < 1000:
            logger.info("Generating synthetic mental health data...")
            data['mental_health'] = self._generate_synthetic_mental_health_data(5000)
        
        return data
    
    def _generate_synthetic_health_data(self, n_samples: int) -> pd.DataFrame:
        """Generate realistic synthetic health data"""
        
        np.random.seed(42)
        
        # Generate user IDs
        user_ids = [f"user_{i:06d}" for i in range(n_samples)]
        
        # Generate demographics
        ages = np.random.normal(45, 15, n_samples).clip(18, 80).astype(int)
        genders = np.random.choice(['male', 'female'], n_samples)
        heights = np.where(genders == 'male', 
                          np.random.normal(175, 7, n_samples),
                          np.random.normal(162, 6, n_samples)).clip(150, 200)
        
        # Generate correlated health metrics
        base_weights = np.where(genders == 'male',
                               np.random.normal(80, 12, n_samples),
                               np.random.normal(65, 10, n_samples)).clip(45, 150)
        
        bmis = base_weights / ((heights / 100) ** 2)
        
        # Generate time series data (30 days per user)
        all_data = []
        
        for i in range(n_samples):
            base_date = datetime.now() - timedelta(days=30)
            
            for day in range(30):
                timestamp = base_date + timedelta(days=day)
                
                # Add some temporal variation
                weight_variation = np.random.normal(0, 0.5)
                current_weight = base_weights[i] + weight_variation
                current_bmi = current_weight / ((heights[i] / 100) ** 2)
                
                # Generate correlated health metrics
                cholesterol_base = 180 + (bmis[i] - 22) * 5 + np.random.normal(0, 20)
                bp_systolic = 110 + (bmis[i] - 22) * 2 + ages[i] * 0.5 + np.random.normal(0, 10)
                bp_diastolic = 70 + (bmis[i] - 22) * 1 + ages[i] * 0.2 + np.random.normal(0, 5)
                glucose = 90 + (bmis[i] - 22) * 2 + np.random.normal(0, 10)
                
                all_data.append({
                    'user_id': user_ids[i],
                    'timestamp': timestamp,
                    'age': ages[i],
                    'gender': genders[i],
                    'height': heights[i],
                    'weight': current_weight,
                    'bmi': current_bmi,
                    'cholesterol_total': cholesterol_base.clip(120, 350),
                    'cholesterol_hdl': (cholesterol_base * 0.3 + np.random.normal(0, 5)).clip(30, 100),
                    'cholesterol_ldl': (cholesterol_base * 0.6 + np.random.normal(0, 10)).clip(50, 250),
                    'blood_pressure_systolic': bp_systolic.clip(90, 180),
                    'blood_pressure_diastolic': bp_diastolic.clip(60, 110),
                    'glucose': glucose.clip(70, 200),
                    'triglycerides': (cholesterol_base * 0.8 + np.random.normal(0, 30)).clip(50, 400),
                    'nutrition_category': np.random.randint(0, 8),
                    'health_risk': np.random.randint(0, 5),
                    'metabolic_rate': np.random.normal(2000, 300).clip(1200, 3500),
                    'caloric_needs': np.random.normal(2200, 400).clip(1400, 3800)
                })
        
        return pd.DataFrame(all_data)
    
    def _generate_synthetic_genomic_data(self, n_samples: int) -> pd.DataFrame:
        """Generate realistic synthetic genomic data"""
        
        np.random.seed(42)
        
        # Important nutrition-related SNPs
        nutrition_snps = [
            'rs1801282', 'rs1801133', 'rs662799', 'rs5918', 'rs1042713',
            'rs1042714', 'rs4680', 'rs1799971', 'rs16969968', 'rs1051730',
            'rs1800497', 'rs6265', 'rs1800629', 'rs361525', 'rs1143634'
        ]
        
        # Generate additional SNPs to reach 1000
        additional_snps = [f'rs{1000000 + i}' for i in range(985)]
        all_snps = nutrition_snps + additional_snps
        
        genomic_data = []
        
        for i in range(n_samples):
            user_id = f"user_{i:06d}"
            snp_data = {'user_id': user_id}
            
            for snp in all_snps:
                # Generate realistic genotype frequencies
                if snp in nutrition_snps:
                    # More variation for important SNPs
                    genotypes = ['0/0', '0/1', '1/1', './.']
                    probabilities = [0.5, 0.3, 0.15, 0.05]
                else:
                    # Less variation for background SNPs
                    genotypes = ['0/0', '0/1', '1/1', './.']
                    probabilities = [0.7, 0.2, 0.08, 0.02]
                
                snp_data[snp] = np.random.choice(genotypes, p=probabilities)
            
            genomic_data.append(snp_data)
        
        return pd.DataFrame(genomic_data)
    
    def _generate_synthetic_mental_health_data(self, n_samples: int) -> pd.DataFrame:
        """Generate realistic synthetic mental health data"""
        
        np.random.seed(42)
        
        mental_health_data = []
        
        for i in range(n_samples):
            user_id = f"user_{i:06d}"
            
            # Generate correlated mental health metrics
            base_stress = np.random.normal(5, 2).clip(1, 10)
            
            mental_health_data.append({
                'user_id': user_id,
                'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 30)),
                'stress_level': int(base_stress),
                'anxiety_score': int((base_stress * 0.8 + np.random.normal(0, 1)).clip(1, 10)),
                'depression_score': int((base_stress * 0.6 + np.random.normal(0, 1.5)).clip(1, 10)),
                'sleep_quality': int((10 - base_stress * 0.7 + np.random.normal(0, 1)).clip(1, 10)),
                'mood_rating': int((8 - base_stress * 0.5 + np.random.normal(0, 1)).clip(1, 10)),
                'cognitive_function': int((9 - base_stress * 0.4 + np.random.normal(0, 1)).clip(1, 10)),
                'social_support': int(np.random.normal(7, 2).clip(1, 10))
            })
        
        return pd.DataFrame(mental_health_data)

class NHANESProcessor:
    """Processor for NHANES health survey data"""
    
    async def process(self) -> pd.DataFrame:
        """Process NHANES data"""
        
        # Check if local NHANES data exists
        nhanes_path = Path(settings.NHANES_DATA_PATH)
        
        if nhanes_path.exists():
            logger.info("Loading local NHANES data...")
            return pd.read_csv(nhanes_path)
        else:
            logger.warning("NHANES data file not found. Using synthetic data.")
            return pd.DataFrame()  # Will trigger synthetic data generation

class GenomicProcessor:
    """Processor for genomic SNP data"""
    
    async def process(self) -> pd.DataFrame:
        """Process genomic data"""
        
        genomic_path = Path(settings.GENOMIC_DATA_PATH)
        
        if genomic_path.exists():
            logger.info("Loading local genomic data...")
            df = pd.read_csv(genomic_path)
            
            # Clean and standardize genomic data
            df = self._clean_genomic_data(df)
            return df
        else:
            logger.warning("Genomic data file not found. Using synthetic data.")
            return pd.DataFrame()
    
    def _clean_genomic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize genomic data"""
        
        # Standardize genotype format
        snp_columns = [col for col in df.columns if col.startswith('rs')]
        
        for col in snp_columns:
            # Convert various genotype formats to standard format
            df[col] = df[col].astype(str)
            df[col] = df[col].replace({
                'AA': '0/0', 'AB': '0/1', 'BA': '1/0', 'BB': '1/1',
                'nan': './.', 'NaN': './.', 'None': './.', '': './.'}
            )
        
        return df

class USDAProcessor:
    """Processor for USDA nutrition database"""
    
    async def process(self) -> pd.DataFrame:
        """Process USDA nutrition data"""
        
        usda_path = Path(settings.USDA_NUTRIENT_DATA_PATH)
        
        if usda_path.exists():
            logger.info("Loading local USDA data...")
            return pd.read_csv(usda_path)
        else:
            logger.info("Fetching USDA nutrition data...")
            return await self._fetch_usda_data()
    
    async def _fetch_usda_data(self) -> pd.DataFrame:
        """Fetch data from USDA FoodData Central API"""
        
        if not settings.USDA_API_KEY:
            logger.warning("USDA API key not configured. Using placeholder data.")
            return self._create_placeholder_usda_data()
        
        # Placeholder for USDA API integration
        # In real implementation, would fetch from https://fdc.nal.usda.gov/
        return self._create_placeholder_usda_data()
    
    def _create_placeholder_usda_data(self) -> pd.DataFrame:
        """Create placeholder USDA nutrition data"""
        
        foods = [
            {
                'fdc_id': '001001', 'description': 'Chicken, broilers or fryers, breast, meat only, cooked, roasted',
                'food_category': 'Poultry Products', 'calories': 165, 'protein': 31.02, 'fat': 3.57,
                'carbohydrate': 0, 'fiber': 0, 'sugars': 0, 'sodium': 74
            },
            {
                'fdc_id': '001002', 'description': 'Salmon, Atlantic, farmed, cooked, dry heat',
                'food_category': 'Finfish and Shellfish Products', 'calories': 206, 'protein': 25.44, 'fat': 12.35,
                'carbohydrate': 0, 'fiber': 0, 'sugars': 0, 'sodium': 59
            },
            # Add more foods...
        ]
        
        return pd.DataFrame(foods)

class DrugBankProcessor:
    """Processor for DrugBank drug-nutrient interaction data"""
    
    async def process(self) -> pd.DataFrame:
        """Process DrugBank interaction data"""
        
        drugbank_path = Path(settings.DRUGBANK_DATA_PATH)
        
        if drugbank_path.exists():
            logger.info("Loading local DrugBank data...")
            return pd.read_csv(drugbank_path)
        else:
            logger.info("Creating placeholder drug-nutrient interaction data...")
            return self._create_placeholder_drugbank_data()
    
    def _create_placeholder_drugbank_data(self) -> pd.DataFrame:
        """Create placeholder drug-nutrient interaction data"""
        
        interactions = [
            {
                'drug_name': 'Warfarin', 'nutrient': 'Vitamin K', 'interaction_type': 'antagonistic',
                'severity': 'high', 'description': 'Vitamin K can reduce warfarin effectiveness',
                'recommendation': 'Maintain consistent vitamin K intake'
            },
            {
                'drug_name': 'Metformin', 'nutrient': 'Vitamin B12', 'interaction_type': 'depletion',
                'severity': 'moderate', 'description': 'Metformin can reduce B12 absorption',
                'recommendation': 'Consider B12 supplementation'
            },
            # Add more interactions...
        ]
        
        return pd.DataFrame(interactions)

class MentalHealthProcessor:
    """Processor for mental health assessment data"""
    
    async def process(self) -> pd.DataFrame:
        """Process mental health data"""
        
        mental_health_path = Path(settings.MENTAL_HEALTH_DATA_PATH)
        
        if mental_health_path.exists():
            logger.info("Loading local mental health data...")
            df = pd.read_csv(mental_health_path)
            return self._clean_mental_health_data(df)
        else:
            logger.warning("Mental health data file not found. Using synthetic data.")
            return pd.DataFrame()
    
    def _clean_mental_health_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate mental health data"""
        
        # Ensure scores are within valid ranges (1-10)
        score_columns = ['stress_level', 'anxiety_score', 'depression_score', 
                        'sleep_quality', 'mood_rating', 'cognitive_function', 'social_support']
        
        for col in score_columns:
            if col in df.columns:
                df[col] = df[col].clip(1, 10)
        
        return df

# Data validation functions
def validate_health_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate health data quality"""
    
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'quality_score': 0.0
    }
    
    # Check for required columns
    required_columns = ['user_id', 'age', 'gender', 'height', 'weight']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        validation_results['errors'].append(f"Missing required columns: {missing_columns}")
        validation_results['is_valid'] = False
    
    # Check data ranges
    if 'age' in df.columns:
        invalid_ages = df[(df['age'] < 0) | (df['age'] > 120)]
        if len(invalid_ages) > 0:
            validation_results['warnings'].append(f"Invalid age values: {len(invalid_ages)} records")
    
    # Calculate quality score
    total_checks = 10
    passed_checks = total_checks - len(validation_results['warnings']) - len(validation_results['errors'])
    validation_results['quality_score'] = max(0.0, passed_checks / total_checks)
    
    return validation_results

def validate_genomic_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate genomic data quality"""
    
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'quality_score': 0.0
    }
    
    # Check SNP format
    snp_columns = [col for col in df.columns if col.startswith('rs')]
    
    if len(snp_columns) == 0:
        validation_results['errors'].append("No SNP columns found")
        validation_results['is_valid'] = False
    
    # Check genotype format
    valid_genotypes = {'0/0', '0/1', '1/0', '1/1', './.'}
    
    for col in snp_columns[:10]:  # Check first 10 SNPs
        invalid_genotypes = df[~df[col].isin(valid_genotypes)]
        if len(invalid_genotypes) > 0:
            validation_results['warnings'].append(f"Invalid genotypes in {col}: {len(invalid_genotypes)} records")
    
    # Calculate quality score
    validation_results['quality_score'] = max(0.0, 1.0 - len(validation_results['warnings']) * 0.1)
    
    return validation_results

# Export main classes and functions
__all__ = [
    'DataPreprocessor',
    'NHANESProcessor',
    'GenomicProcessor', 
    'USDAProcessor',
    'DrugBankProcessor',
    'MentalHealthProcessor',
    'validate_health_data',
    'validate_genomic_data'
]
