"""
AI-Powered Personalized Nutrition Planning System
Main FastAPI application entry point
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager
import logging
from typing import Dict, Any

from api.routes import health, genomics, nutrition, recommendations
from core.config import settings
from core.database import init_db
from models.ai_models import ModelManager
from utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global model manager instance
model_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global model_manager
    
    # Startup
    logger.info("Starting AI-Powered Nutrition Planning System...")
    
    # Initialize database
    await init_db()
    
    # Initialize AI models
    model_manager = ModelManager()
    await model_manager.load_models()
    
    # Store model manager in app state
    app.state.model_manager = model_manager
    
    logger.info("System startup complete!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down system...")
    if model_manager:
        await model_manager.cleanup()

# Create FastAPI app
app = FastAPI(
    title="AI-Powered Personalized Nutrition Planning System",
    description="Advanced nutrition planning using LSTM, Transformer, and Fusion models with genomic, health, and mental health data integration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(health.router, prefix="/api/v1/health", tags=["Health"])
app.include_router(genomics.router, prefix="/api/v1/genomics", tags=["Genomics"])
app.include_router(nutrition.router, prefix="/api/v1/nutrition", tags=["Nutrition"])
app.include_router(recommendations.router, prefix="/api/v1/recommendations", tags=["Recommendations"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI-Powered Personalized Nutrition Planning System",
        "version": "1.0.0",
        "status": "active",
        "features": [
            "LSTM-based health data analysis",
            "Transformer-based genomic analysis", 
            "Fusion model recommendations",
            "Real-time adaptation",
            "Multi-modal data integration"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": "2024-09-20T00:59:16+05:30"}

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

def get_model_manager() -> ModelManager:
    """Dependency to get model manager"""
    return app.state.model_manager

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
