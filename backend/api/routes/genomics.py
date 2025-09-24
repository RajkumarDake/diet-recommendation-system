"""
Genomics API Routes
Endpoints for genomic data analysis and Transformer model predictions
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import logging
from io import StringIO
import json

from models.ai_models import ModelManager

logger = logging.getLogger(__name__)
router = APIRouter()

class SNPData(BaseModel):
    """Single SNP data point"""
    snp_id: str = Field(..., pattern="^rs[0-9]+$")
    chromosome: Optional[str] = None
    position: Optional[int] = None
    genotype: str = Field(..., pattern="^(0/0|0/1|1/0|1/1|\\./\\.)$")
    allele_frequency: Optional[float] = Field(None, ge=0.0, le=1.0)

class GenomicProfile(BaseModel):
    """Complete genomic profile"""
    user_id: str
    snps: List[SNPData]
    ancestry: Optional[str] = None
    sequencing_platform: Optional[str] = None
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)

class GenomicPrediction(BaseModel):
    """Genomic analysis prediction response"""
    metabolism_efficiency: Dict[str, Any]
    absorption_capacity: Dict[str, Any]
    sensitivity_risk: Dict[str, Any]
    inflammation_response: Dict[str, Any]
    vitamin_d_synthesis: float
    caffeine_metabolism: float
    nutrient_recommendations: List[Dict[str, Any]]
    genetic_insights: Dict[str, Any]

class DrugNutrientInteraction(BaseModel):
    """Drug-nutrient interaction data"""
    drug_name: str
    nutrient: str
    interaction_type: str
    severity: str
    description: str
    recommendation: str

def get_model_manager() -> ModelManager:
    """Dependency to get model manager"""
    from main import app
    return app.state.model_manager

@router.post("/analyze", response_model=GenomicPrediction)
async def analyze_genomic_data(
    genomic_profile: GenomicProfile,
    model_manager: ModelManager = Depends(get_model_manager)
) -> GenomicPrediction:
    """
    Analyze genomic data using Transformer model
    
    Processes SNP variations to predict nutrient metabolism,
    absorption capacity, and genetic predispositions.
    """
    try:
        logger.info(f"Analyzing genomic data for user: {genomic_profile.user_id}")
        
        # Validate model availability
        status = await model_manager.get_model_status()
        if not status['transformer_model']['trained']:
            raise HTTPException(
                status_code=503,
                detail="Transformer genomic model not available. Please train model first."
            )
        
        # Prepare genomic data
        genomic_df = _prepare_genomic_dataframe(genomic_profile)
        
        # Get predictions
        predictions = await model_manager.get_genomic_predictions(genomic_df)
        
        # Generate nutrient recommendations
        nutrient_recommendations = _generate_nutrient_recommendations(predictions, genomic_profile)
        
        # Generate genetic insights
        genetic_insights = _generate_genetic_insights(genomic_profile, predictions)
        
        return GenomicPrediction(
            metabolism_efficiency=predictions['metabolism_efficiency'],
            absorption_capacity=predictions['absorption_capacity'],
            sensitivity_risk=predictions['sensitivity_risk'],
            inflammation_response=predictions['inflammation_response'],
            vitamin_d_synthesis=predictions['vitamin_d_synthesis'],
            caffeine_metabolism=predictions['caffeine_metabolism'],
            nutrient_recommendations=nutrient_recommendations,
            genetic_insights=genetic_insights
        )
        
    except Exception as e:
        logger.error(f"Error analyzing genomic data: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing genomic data: {str(e)}")

@router.post("/upload-vcf")
async def upload_vcf_file(
    user_id: str,
    file: UploadFile = File(...),
    model_manager: ModelManager = Depends(get_model_manager)
) -> Dict[str, Any]:
    """
    Upload and process VCF (Variant Call Format) file
    
    Accepts genomic data in standard VCF format and processes it
    for nutrition analysis.
    """
    try:
        if not file.filename.endswith(('.vcf', '.vcf.gz')):
            raise HTTPException(status_code=400, detail="File must be in VCF format")
        
        # Read VCF file content
        content = await file.read()
        vcf_content = content.decode('utf-8')
        
        # Parse VCF file
        snp_data = _parse_vcf_content(vcf_content)
        
        # Create genomic profile
        genomic_profile = GenomicProfile(
            user_id=user_id,
            snps=snp_data,
            sequencing_platform="uploaded_vcf"
        )
        
        # Analyze genomic data
        analysis_result = await analyze_genomic_data(genomic_profile, model_manager)
        
        return {
            "status": "success",
            "user_id": user_id,
            "snps_processed": len(snp_data),
            "analysis": analysis_result.dict(),
            "upload_timestamp": "2025-09-20T01:10:53+05:30"
        }
        
    except Exception as e:
        logger.error(f"Error processing VCF file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing VCF file: {str(e)}")

@router.get("/snp-info/{snp_id}")
async def get_snp_information(snp_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific SNP
    
    Provides information about SNP function, associated traits,
    and nutritional implications.
    """
    try:
        # SNP information database (would be loaded from external sources)
        snp_database = {
            "rs1801282": {
                "gene": "PPARG",
                "function": "Peroxisome proliferator-activated receptor gamma",
                "nutritional_impact": "Fat metabolism and insulin sensitivity",
                "dietary_recommendations": ["Omega-3 fatty acids", "Complex carbohydrates"],
                "associated_traits": ["Type 2 diabetes risk", "Obesity susceptibility"]
            },
            "rs1801133": {
                "gene": "MTHFR",
                "function": "Methylenetetrahydrofolate reductase",
                "nutritional_impact": "Folate metabolism and homocysteine levels",
                "dietary_recommendations": ["Folate-rich foods", "B-vitamin complex"],
                "associated_traits": ["Cardiovascular disease risk", "Neural tube defects"]
            },
            "rs662799": {
                "gene": "APOA5",
                "function": "Apolipoprotein A5",
                "nutritional_impact": "Triglyceride metabolism",
                "dietary_recommendations": ["Limit saturated fats", "Increase fiber"],
                "associated_traits": ["Hypertriglyceridemia", "Cardiovascular disease"]
            }
        }
        
        if snp_id not in snp_database:
            # Fetch from external API (placeholder)
            return {
                "snp_id": snp_id,
                "status": "not_found",
                "message": "SNP information not available in current database",
                "suggestion": "Consider consulting dbSNP database for more information"
            }
        
        snp_info = snp_database[snp_id]
        snp_info["snp_id"] = snp_id
        snp_info["clinical_significance"] = _get_clinical_significance(snp_id)
        
        return snp_info
        
    except Exception as e:
        logger.error(f"Error getting SNP information: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting SNP info: {str(e)}")

@router.post("/drug-interactions")
async def check_drug_nutrient_interactions(
    genomic_profile: GenomicProfile,
    medications: List[str]
) -> List[DrugNutrientInteraction]:
    """
    Check for drug-nutrient interactions based on genetic profile
    
    Analyzes genetic variants that affect drug metabolism and
    identifies potential nutrient interactions.
    """
    try:
        interactions = []
        
        # Drug-nutrient interaction database (simplified)
        interaction_db = {
            "warfarin": {
                "affected_snps": ["rs9923231", "rs1799853"],
                "nutrients": ["Vitamin K", "Vitamin E"],
                "interactions": [
                    {
                        "nutrient": "Vitamin K",
                        "interaction_type": "antagonistic",
                        "severity": "high",
                        "description": "Vitamin K can reduce warfarin effectiveness",
                        "recommendation": "Maintain consistent vitamin K intake"
                    }
                ]
            },
            "metformin": {
                "affected_snps": ["rs11212617", "rs8192675"],
                "nutrients": ["Vitamin B12", "Folate"],
                "interactions": [
                    {
                        "nutrient": "Vitamin B12",
                        "interaction_type": "depletion",
                        "severity": "moderate",
                        "description": "Metformin can reduce B12 absorption",
                        "recommendation": "Consider B12 supplementation"
                    }
                ]
            }
        }
        
        # Check each medication
        for medication in medications:
            if medication.lower() in interaction_db:
                drug_info = interaction_db[medication.lower()]
                
                # Check if user has relevant SNPs
                user_snps = {snp.snp_id for snp in genomic_profile.snps}
                relevant_snps = set(drug_info["affected_snps"]) & user_snps
                
                if relevant_snps:
                    for interaction in drug_info["interactions"]:
                        interactions.append(DrugNutrientInteraction(
                            drug_name=medication,
                            nutrient=interaction["nutrient"],
                            interaction_type=interaction["interaction_type"],
                            severity=interaction["severity"],
                            description=interaction["description"],
                            recommendation=interaction["recommendation"]
                        ))
        
        return interactions
        
    except Exception as e:
        logger.error(f"Error checking drug interactions: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking interactions: {str(e)}")

@router.get("/ancestry-nutrition/{ancestry}")
async def get_ancestry_nutrition_insights(ancestry: str) -> Dict[str, Any]:
    """
    Get nutrition insights based on genetic ancestry
    
    Provides population-specific nutritional recommendations
    based on genetic ancestry patterns.
    """
    try:
        ancestry_insights = {
            "european": {
                "lactose_tolerance": "high",
                "alcohol_metabolism": "variable",
                "vitamin_d_synthesis": "low_in_northern_populations",
                "recommendations": [
                    "Higher vitamin D supplementation may be needed",
                    "Generally good lactose tolerance",
                    "Mediterranean diet patterns beneficial"
                ]
            },
            "east_asian": {
                "lactose_tolerance": "low",
                "alcohol_metabolism": "reduced_aldh2_activity",
                "folate_metabolism": "efficient",
                "recommendations": [
                    "Limit dairy products or use lactase supplements",
                    "Moderate alcohol consumption",
                    "Rich in folate-containing foods beneficial"
                ]
            },
            "african": {
                "lactose_tolerance": "variable",
                "vitamin_d_synthesis": "efficient",
                "salt_sensitivity": "higher_prevalence",
                "recommendations": [
                    "Lower sodium intake recommended",
                    "Generally efficient vitamin D synthesis",
                    "Traditional plant-based diets beneficial"
                ]
            }
        }
        
        if ancestry.lower() not in ancestry_insights:
            return {
                "ancestry": ancestry,
                "status": "limited_data",
                "message": "Limited ancestry-specific data available",
                "general_recommendations": [
                    "Focus on whole foods",
                    "Maintain balanced macronutrients",
                    "Consider individual genetic testing"
                ]
            }
        
        insights = ancestry_insights[ancestry.lower()]
        insights["ancestry"] = ancestry
        insights["population_studies"] = _get_population_studies(ancestry)
        
        return insights
        
    except Exception as e:
        logger.error(f"Error getting ancestry insights: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting ancestry insights: {str(e)}")

@router.post("/validate-genomic")
async def validate_genomic_data(
    genomic_profile: GenomicProfile
) -> Dict[str, Any]:
    """
    Validate genomic data for quality and completeness
    
    Checks for data quality issues, missing SNPs, and
    provides quality metrics.
    """
    try:
        validation_results = {
            "is_valid": True,
            "quality_score": 0.0,
            "completeness_score": 0.0,
            "warnings": [],
            "errors": [],
            "snp_coverage": {}
        }
        
        # Check SNP coverage for nutrition-related genes
        nutrition_snps = {
            "metabolism": ["rs1801282", "rs1801133", "rs662799", "rs5918"],
            "absorption": ["rs1042713", "rs1042714", "rs4680", "rs1799971"],
            "sensitivity": ["rs16969968", "rs1051730", "rs1800497", "rs6265"]
        }
        
        user_snps = {snp.snp_id for snp in genomic_profile.snps}
        
        for category, required_snps in nutrition_snps.items():
            covered_snps = len(set(required_snps) & user_snps)
            coverage_percent = covered_snps / len(required_snps)
            validation_results["snp_coverage"][category] = {
                "covered": covered_snps,
                "total": len(required_snps),
                "percentage": coverage_percent
            }
            
            if coverage_percent < 0.5:
                validation_results["warnings"].append(
                    f"Low coverage for {category} SNPs ({coverage_percent:.1%})"
                )
        
        # Check for invalid genotypes
        invalid_genotypes = []
        for snp in genomic_profile.snps:
            if snp.genotype not in ["0/0", "0/1", "1/0", "1/1", "./."]:
                invalid_genotypes.append(snp.snp_id)
        
        if invalid_genotypes:
            validation_results["errors"].append(f"Invalid genotypes: {', '.join(invalid_genotypes)}")
            validation_results["is_valid"] = False
        
        # Calculate overall scores
        total_coverage = np.mean([cat["percentage"] for cat in validation_results["snp_coverage"].values()])
        validation_results["completeness_score"] = total_coverage
        validation_results["quality_score"] = max(0.0, 1.0 - (len(validation_results["warnings"]) * 0.1))
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Error validating genomic data: {e}")
        raise HTTPException(status_code=500, detail=f"Error validating genomic data: {str(e)}")

# Helper functions
def _prepare_genomic_dataframe(genomic_profile: GenomicProfile) -> pd.DataFrame:
    """Convert genomic profile to DataFrame for model input"""
    
    snp_dict = {}
    for snp in genomic_profile.snps:
        snp_dict[snp.snp_id] = snp.genotype
    
    # Ensure we have data for required SNPs (pad with missing if necessary)
    required_snps = [f"rs{i}" for i in range(1000)]  # Placeholder
    for snp_id in required_snps:
        if snp_id not in snp_dict:
            snp_dict[snp_id] = "./."  # Missing genotype
    
    return pd.DataFrame([snp_dict])

def _parse_vcf_content(vcf_content: str) -> List[SNPData]:
    """Parse VCF file content and extract SNP data"""
    
    snp_data = []
    lines = vcf_content.split('\n')
    
    for line in lines:
        if line.startswith('#') or not line.strip():
            continue
        
        fields = line.split('\t')
        if len(fields) >= 10:  # Standard VCF format
            chrom = fields[0]
            pos = int(fields[1])
            snp_id = fields[2] if fields[2] != '.' else f"chr{chrom}_{pos}"
            ref = fields[3]
            alt = fields[4]
            genotype_field = fields[9].split(':')[0]  # First sample genotype
            
            # Convert genotype format
            if '/' in genotype_field:
                genotype = genotype_field
            elif '|' in genotype_field:
                genotype = genotype_field.replace('|', '/')
            else:
                genotype = "./."
            
            if snp_id.startswith('rs'):  # Only include rsIDs
                snp_data.append(SNPData(
                    snp_id=snp_id,
                    chromosome=chrom,
                    position=pos,
                    genotype=genotype
                ))
    
    return snp_data[:1000]  # Limit to first 1000 SNPs

def _generate_nutrient_recommendations(predictions: Dict[str, Any], genomic_profile: GenomicProfile) -> List[Dict[str, Any]]:
    """Generate nutrient recommendations based on genomic predictions"""
    
    recommendations = []
    
    # Metabolism efficiency recommendations
    metabolism_class = predictions['metabolism_efficiency']['predicted_class']
    if metabolism_class <= 2:  # Low efficiency
        recommendations.append({
            "nutrient": "B-Complex Vitamins",
            "reason": "Support metabolic pathways",
            "dosage": "As per RDA",
            "confidence": predictions['metabolism_efficiency']['confidence']
        })
    
    # Absorption capacity recommendations
    absorption_class = predictions['absorption_capacity']['predicted_class']
    if absorption_class <= 2:  # Low absorption
        recommendations.append({
            "nutrient": "Digestive Enzymes",
            "reason": "Improve nutrient absorption",
            "dosage": "With meals",
            "confidence": predictions['absorption_capacity']['confidence']
        })
    
    # Vitamin D synthesis
    if predictions['vitamin_d_synthesis'] < 0.5:
        recommendations.append({
            "nutrient": "Vitamin D3",
            "reason": "Low genetic synthesis capacity",
            "dosage": "1000-2000 IU daily",
            "confidence": predictions['vitamin_d_synthesis']
        })
    
    # Caffeine metabolism
    if predictions['caffeine_metabolism'] < 0.3:
        recommendations.append({
            "nutrient": "Limit Caffeine",
            "reason": "Slow caffeine metabolism",
            "dosage": "< 200mg daily",
            "confidence": predictions['caffeine_metabolism']
        })
    
    return recommendations

def _generate_genetic_insights(genomic_profile: GenomicProfile, predictions: Dict[str, Any]) -> Dict[str, Any]:
    """Generate genetic insights from genomic analysis"""
    
    insights = {
        "metabolic_profile": "",
        "nutrient_sensitivities": [],
        "dietary_adaptations": [],
        "risk_factors": []
    }
    
    # Metabolic profile
    metabolism_class = predictions['metabolism_efficiency']['predicted_class']
    if metabolism_class >= 4:
        insights["metabolic_profile"] = "Efficient metabolizer - can handle varied diet"
    elif metabolism_class >= 2:
        insights["metabolic_profile"] = "Moderate metabolizer - balanced approach needed"
    else:
        insights["metabolic_profile"] = "Slow metabolizer - careful nutrient timing required"
    
    # Sensitivity analysis
    sensitivity_class = predictions['sensitivity_risk']['predicted_class']
    if sensitivity_class >= 2:
        insights["nutrient_sensitivities"].extend(["Gluten sensitivity risk", "Dairy sensitivity risk"])
    
    # Dietary adaptations
    if predictions['vitamin_d_synthesis'] < 0.5:
        insights["dietary_adaptations"].append("Increase vitamin D rich foods")
    
    if predictions['caffeine_metabolism'] < 0.3:
        insights["dietary_adaptations"].append("Limit caffeine intake")
    
    # Risk factors
    inflammation_class = predictions['inflammation_response']['predicted_class']
    if inflammation_class >= 1:
        insights["risk_factors"].append("Elevated inflammation response")
    
    return insights

def _get_clinical_significance(snp_id: str) -> str:
    """Get clinical significance of SNP"""
    # Placeholder - would query clinical databases
    significance_map = {
        "rs1801282": "Pathogenic/Likely pathogenic",
        "rs1801133": "Uncertain significance",
        "rs662799": "Benign/Likely benign"
    }
    return significance_map.get(snp_id, "Unknown")

def _get_population_studies(ancestry: str) -> List[Dict[str, str]]:
    """Get relevant population studies for ancestry"""
    # Placeholder for population study references
    return [
        {
            "study": f"{ancestry.title()} Nutrition Genomics Study",
            "year": "2023",
            "finding": "Population-specific nutrient requirements identified"
        }
    ]
