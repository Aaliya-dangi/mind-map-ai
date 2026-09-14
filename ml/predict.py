"""
MindMap AI - Live Prediction & Explainability Pipeline
Executes live model inference on single or batch student inputs, computes continuous
burnout risk scores, and extracts per-prediction explainable contributing factors
per design.md §22-23 and RMD.md §4, §11-12.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import numpy as np
import pandas as pd
import joblib

# Add project root to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from features.definitions import (
    RAW_NUMERICAL_FEATURES,
    DERIVED_NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_METADATA,
)
from features.pipeline import enrich_features
from features.engineering import assign_risk_category

# Cache loaded artifacts
_PREPROCESSOR = None
_MODEL = None
_BASELINE_STATS = None


def load_artifacts():
    """Loads and caches persisted model, preprocessor, and baseline statistics."""
    global _PREPROCESSOR, _MODEL, _BASELINE_STATS

    if _PREPROCESSOR is None or _MODEL is None:
        if not config.PREPROCESSOR_FILE.exists() or not config.BEST_MODEL_FILE.exists():
            from ml.train import train_and_evaluate_models
            print("[INFO] Model artifacts not found. Initiating training...")
            train_and_evaluate_models()

        _PREPROCESSOR = joblib.load(config.PREPROCESSOR_FILE)
        _MODEL = joblib.load(config.BEST_MODEL_FILE)

    if _BASELINE_STATS is None:
        if config.PROCESSED_DATA_FILE.exists():
            df_clean = pd.read_csv(config.PROCESSED_DATA_FILE)
            _BASELINE_STATS = {
                col: {"mean": float(df_clean[col].mean()), "std": float(df_clean[col].std()) or 1.0}
                for col in (RAW_NUMERICAL_FEATURES + DERIVED_NUMERICAL_FEATURES)
                if col in df_clean.columns
            }
        else:
            _BASELINE_STATS = {}

    return _PREPROCESSOR, _MODEL, _BASELINE_STATS


def extract_top_contributing_factors(
    input_row: pd.Series,
    top_n: int = 4,
) -> List[Dict[str, Any]]:
    """
    Extracts top features influencing this specific individual's risk score.
    Phrased strictly as model influence, avoiding causal claims (design.md §22).
    """
    _, model, baseline_stats = load_artifacts()

    contributions = []

    # Direction of influence by feature nature
    risk_increasing_features = {
        "assignment_workload",
        "academic_pressure",
        "study_hours_per_day",
        "exam_frequency",
        "screen_time_hours",
        "academic_pressure_score",
    }
    protective_features = {
        "sleep_hours",
        "sleep_quality",
        "physical_activity_hours",
        "social_interaction_hours",
        "breaks_per_day",
        "hobbies_hours_per_week",
        "days_off_per_week",
        "lifestyle_balance_score",
        "attendance_percentage",
    }

    for feat in (RAW_NUMERICAL_FEATURES + DERIVED_NUMERICAL_FEATURES):
        if feat not in input_row or feat not in baseline_stats:
            continue

        val = float(input_row[feat])
        mean = baseline_stats[feat]["mean"]
        std = baseline_stats[feat]["std"]
        z_score = (val - mean) / std

        # Weight by feature importance/impact
        impact = abs(z_score)
        if feat in ["academic_pressure_score", "lifestyle_balance_score"]:
            impact *= 1.5

        meta = FEATURE_METADATA.get(feat, {"label": feat.replace("_", " ").title(), "unit": ""})
        label = meta["label"]
        unit = meta.get("unit", "")

        if feat in risk_increasing_features:
            if z_score > 0.25:
                direction = "elevates_risk"
                explanation = f"Higher {label} ({val} {unit}) compared to cohort average ({mean:.1f} {unit}) contributed to elevated risk."
            elif z_score < -0.25:
                direction = "lowers_risk"
                explanation = f"Lower {label} ({val} {unit}) compared to cohort average ({mean:.1f} {unit}) helped mitigate pressure."
            else:
                continue
        elif feat in protective_features:
            if z_score < -0.25:
                direction = "elevates_risk"
                explanation = f"Below-average {label} ({val} {unit}) compared to cohort average ({mean:.1f} {unit}) reduced protective lifestyle buffer."
            elif z_score > 0.25:
                direction = "lowers_risk"
                explanation = f"Above-average {label} ({val} {unit}) compared to cohort average ({mean:.1f} {unit}) served as a strong positive buffer."
            else:
                continue
        else:
            continue

        contributions.append({
            "feature": feat,
            "label": label,
            "value": val,
            "unit": unit,
            "cohort_avg": round(mean, 1),
            "direction": direction,
            "impact_magnitude": round(impact, 3),
            "explanation": explanation,
        })

    # Sort by absolute impact magnitude
    contributions.sort(key=lambda x: x["impact_magnitude"], reverse=True)
    return contributions[:top_n]


def predict_burnout_risk(input_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
    """
    Executes live model inference from user input.

    Parameters:
    - input_data: dictionary of student parameters or single-row DataFrame.

    Returns:
    - Dict with predicted_category, risk_score (0-100), academic/lifestyle scores,
      class probabilities, top contributing factors, and standard disclaimer.
    """
    preprocessor, model, _ = load_artifacts()

    # Step 1: Feature Enrichment
    df_enriched = enrich_features(input_data)
    row = df_enriched.iloc[0]

    # Step 2: Preprocessing
    feature_cols = RAW_NUMERICAL_FEATURES + DERIVED_NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X_input = df_enriched[feature_cols]
    X_trans = preprocessor.transform(X_input)

    # Step 3: Model Prediction
    predicted_class = str(model.predict(X_trans)[0])

    # Class Probabilities
    classes = list(model.classes_)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_trans)[0]
        prob_dict = {cls_name: round(float(p), 4) for cls_name, p in zip(classes, probs)}
        
        # Compute continuous risk score (0-100) from weighted class probabilities
        class_weights = {"Low": 15.0, "Moderate": 45.0, "High": 72.0, "Very High": 90.0}
        score_val = sum(prob_dict.get(c, 0.0) * class_weights.get(c, 50.0) for c in classes)
        risk_score = round(float(np.clip(score_val, 0.0, 100.0)), 1)
    else:
        prob_dict = {c: (1.0 if c == predicted_class else 0.0) for c in classes}
        default_score_map = {"Low": 20.0, "Moderate": 45.0, "High": 70.0, "Very High": 90.0}
        risk_score = default_score_map.get(predicted_class, 50.0)

    # Reconcile risk category with calibrated score if needed
    category = assign_risk_category(risk_score)

    # Step 4: Explainability (Top Contributing Factors)
    contributing_factors = extract_top_contributing_factors(row, top_n=4)

    return {
        "predicted_category": category,
        "risk_score": risk_score,
        "academic_pressure_score": round(float(row["academic_pressure_score"]), 1),
        "lifestyle_balance_score": round(float(row["lifestyle_balance_score"]), 1),
        "probabilities": prob_dict,
        "contributing_factors": contributing_factors,
        "disclaimer": config.NON_MEDICAL_DISCLAIMER,
    }
