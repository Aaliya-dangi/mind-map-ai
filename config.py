"""
MindMap AI - Configuration Module
Centralized configuration for paths, database settings, and project constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env if present
load_dotenv(BASE_DIR / ".env")

# Directory Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ML_DIR = BASE_DIR / "ml"
MODELS_DIR = ML_DIR / "models"
DB_DIR = BASE_DIR / "db"
APP_DIR = BASE_DIR / "app"
FEATURES_DIR = BASE_DIR / "features"
UTILS_DIR = BASE_DIR / "utils"

# File Paths
RAW_DATA_FILE = RAW_DATA_DIR / "student_burnout_raw.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "student_burnout_cleaned.csv"
BEST_MODEL_FILE = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"
MODEL_METRICS_FILE = MODELS_DIR / "model_metrics.json"

# Database Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "mindmap_ai")
DATA_FALLBACK_TO_CSV = os.getenv("DATA_FALLBACK_TO_CSV", "true").lower() in ("true", "1", "yes")

# Risk Category Thresholds (per design.md §23)
RISK_CATEGORIES = {
    "Low": (0.0, 30.0),
    "Moderate": (30.1, 60.0),
    "High": (60.1, 80.0),
    "Very High": (80.1, 100.0),
}

# Standard Disclaimer
NON_MEDICAL_DISCLAIMER = (
    "MindMap AI is an educational and portfolio analytics platform. "
    "Predictions and risk scores are statistical estimates for pattern analysis "
    "and do NOT constitute a medical, psychiatric, or clinical diagnosis."
)

SIMULATOR_DISCLAIMER = (
    "This is a model-based simulation reflecting statistical associations in the training dataset, "
    "not a guaranteed real-world outcome."
)
