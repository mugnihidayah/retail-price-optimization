"""
Configuration Module for Price Optimization System
Centralized configuration with environment variable support
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Base paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass
class ModelConfig:
    """Configuration for ML model training"""
    n_estimators: int = 100
    max_depth: int = 15
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    cv_folds: int = 5
    random_state: int = 42
    
    # Hyperparameter tuning ranges
    tune_n_estimators: tuple = (50, 100, 200)
    tune_max_depth: tuple = (8, 12, 15, 20)
    tune_learning_rate: tuple = (0.01, 0.05, 0.1, 0.2)


@dataclass
class OptimizationConfig:
    """Configuration for price optimization"""
    default_price_range: float = 0.20
    min_price_range: float = 0.05
    max_price_range: float = 0.50
    num_price_points: int = 50
    min_quantity: float = 0.0
    confidence_level: float = 0.95  # For confidence intervals


@dataclass  
class AppConfig:
    """Configuration for Streamlit application"""
    page_title: str = "Retail Pricing Intelligence"
    page_icon: str = "💰"
    layout: str = "wide"
    
    # Paths (from environment or defaults)
    data_path: str = os.getenv(
        "PRICING_DATA_PATH", 
        str(DATA_DIR / "processed" / "train_data.csv")
    )
    model_path: str = os.getenv(
        "PRICING_MODEL_PATH",
        str(MODELS_DIR / "pricing_model.pkl")
    )
    enhanced_model_path: str = os.getenv(
        "PRICING_ENHANCED_MODEL_PATH",
        str(MODELS_DIR / "pricing_model_enhanced.pkl")
    )


# Global config instances
model_config = ModelConfig()
optimization_config = OptimizationConfig()
app_config = AppConfig()


def get_data_path() -> Path:
    """Get data file path with validation"""
    path = Path(app_config.data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return path


def get_model_path(enhanced: bool = True) -> Path:
    """Get model file path"""
    if enhanced:
        path = Path(app_config.enhanced_model_path)
        if path.exists():
            return path
    return Path(app_config.model_path)


def validate_config() -> dict:
    """Validate configuration and return status"""
    status = {
        "data_exists": Path(app_config.data_path).exists(),
        "model_exists": Path(app_config.model_path).exists(),
        "enhanced_model_exists": Path(app_config.enhanced_model_path).exists(),
        "project_root": str(PROJECT_ROOT),
    }
    return status
