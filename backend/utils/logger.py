"""
Logging configuration for the AI-Powered Nutrition System
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any

from core.config import settings

def setup_logging():
    """Setup comprehensive logging configuration"""
    
    # Create logs directory
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for general logs
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "nutrition_system.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "errors.log",
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # Model training handler
    model_handler = logging.handlers.RotatingFileHandler(
        log_dir / "model_training.log",
        maxBytes=20*1024*1024,  # 20MB
        backupCount=3
    )
    model_handler.setLevel(logging.INFO)
    model_handler.setFormatter(detailed_formatter)
    
    # Add model handler to specific loggers
    model_loggers = [
        'models.lstm_health_model',
        'models.transformer_genomic_model', 
        'models.fusion_model',
        'models.ai_models'
    ]
    
    for logger_name in model_loggers:
        logger = logging.getLogger(logger_name)
        logger.addHandler(model_handler)
    
    # API request handler
    api_handler = logging.handlers.RotatingFileHandler(
        log_dir / "api_requests.log",
        maxBytes=15*1024*1024,  # 15MB
        backupCount=5
    )
    api_handler.setLevel(logging.INFO)
    api_handler.setFormatter(detailed_formatter)
    
    # Add API handler to route loggers
    api_loggers = [
        'api.routes.recommendations',
        'api.routes.health',
        'api.routes.genomics',
        'api.routes.nutrition'
    ]
    
    for logger_name in api_loggers:
        logger = logging.getLogger(logger_name)
        logger.addHandler(api_handler)

class StructuredLogger:
    """Structured logger for better log analysis"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def log_model_training(self, model_type: str, epoch: int, metrics: Dict[str, float]):
        """Log model training progress"""
        log_data = {
            "event_type": "model_training",
            "model_type": model_type,
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(log_data))
    
    def log_prediction(self, model_type: str, user_id: str, confidence: float, processing_time: float):
        """Log model predictions"""
        log_data = {
            "event_type": "prediction",
            "model_type": model_type,
            "user_id": user_id,
            "confidence": confidence,
            "processing_time_ms": processing_time * 1000,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(log_data))
    
    def log_api_request(self, endpoint: str, method: str, user_id: str, response_time: float, status_code: int):
        """Log API requests"""
        log_data = {
            "event_type": "api_request",
            "endpoint": endpoint,
            "method": method,
            "user_id": user_id,
            "response_time_ms": response_time * 1000,
            "status_code": status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(log_data))
    
    def log_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None):
        """Log structured errors"""
        log_data = {
            "event_type": "error",
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.error(json.dumps(log_data))
    
    def log_user_feedback(self, user_id: str, recommendation_id: str, rating: int, feedback_type: str):
        """Log user feedback"""
        log_data = {
            "event_type": "user_feedback",
            "user_id": user_id,
            "recommendation_id": recommendation_id,
            "rating": rating,
            "feedback_type": feedback_type,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(log_data))

# Performance monitoring
class PerformanceLogger:
    """Logger for performance monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger("performance")
        
        # Performance log handler
        perf_handler = logging.handlers.RotatingFileHandler(
            Path("./logs") / "performance.log",
            maxBytes=10*1024*1024,
            backupCount=3
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(logging.Formatter(
            '%(asctime)s - PERFORMANCE - %(message)s'
        ))
        self.logger.addHandler(perf_handler)
        self.logger.setLevel(logging.INFO)
    
    def log_model_performance(self, model_type: str, metrics: Dict[str, float]):
        """Log model performance metrics"""
        perf_data = {
            "metric_type": "model_performance",
            "model_type": model_type,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(perf_data))
    
    def log_system_metrics(self, cpu_usage: float, memory_usage: float, disk_usage: float):
        """Log system resource usage"""
        perf_data = {
            "metric_type": "system_resources",
            "cpu_usage_percent": cpu_usage,
            "memory_usage_percent": memory_usage,
            "disk_usage_percent": disk_usage,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(perf_data))
    
    def log_api_performance(self, endpoint: str, response_time: float, throughput: float):
        """Log API performance metrics"""
        perf_data = {
            "metric_type": "api_performance",
            "endpoint": endpoint,
            "response_time_ms": response_time * 1000,
            "throughput_rps": throughput,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(perf_data))

# Security logging
class SecurityLogger:
    """Logger for security events"""
    
    def __init__(self):
        self.logger = logging.getLogger("security")
        
        # Security log handler
        security_handler = logging.handlers.RotatingFileHandler(
            Path("./logs") / "security.log",
            maxBytes=5*1024*1024,
            backupCount=5
        )
        security_handler.setLevel(logging.WARNING)
        security_handler.setFormatter(logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(security_handler)
        self.logger.setLevel(logging.WARNING)
    
    def log_authentication_attempt(self, user_id: str, success: bool, ip_address: str):
        """Log authentication attempts"""
        security_data = {
            "event_type": "authentication",
            "user_id": user_id,
            "success": success,
            "ip_address": ip_address,
            "timestamp": datetime.utcnow().isoformat()
        }
        level = logging.INFO if success else logging.WARNING
        self.logger.log(level, json.dumps(security_data))
    
    def log_data_access(self, user_id: str, data_type: str, action: str):
        """Log sensitive data access"""
        security_data = {
            "event_type": "data_access",
            "user_id": user_id,
            "data_type": data_type,
            "action": action,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.info(json.dumps(security_data))
    
    def log_suspicious_activity(self, description: str, user_id: str = None, ip_address: str = None):
        """Log suspicious activities"""
        security_data = {
            "event_type": "suspicious_activity",
            "description": description,
            "user_id": user_id,
            "ip_address": ip_address,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logger.warning(json.dumps(security_data))

# Initialize loggers
structured_logger = StructuredLogger("nutrition_system")
performance_logger = PerformanceLogger()
security_logger = SecurityLogger()

# Export commonly used loggers
__all__ = [
    'setup_logging',
    'StructuredLogger',
    'PerformanceLogger', 
    'SecurityLogger',
    'structured_logger',
    'performance_logger',
    'security_logger'
]
