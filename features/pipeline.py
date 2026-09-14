"""
MindMap AI - Feature Pipeline & Transformation
Prepares full feature vectors (raw + derived) from input dictionaries or DataFrames
for downstream ML inference and what-if simulation.
"""

from typing import Dict, Any, Union
import pandas as pd
import numpy as np

from features.engineering import (
    calculate_academic_pressure_score,
    calculate_lifestyle_balance_score,
    assign_risk_category,
)
from features.definitions import ALL_PREDICTOR_FEATURES


def enrich_features(data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
    """
    Takes raw student features (as a dictionary or DataFrame) and computes
    derived composite features: academic_pressure_score and lifestyle_balance_score.

    Returns:
    - pd.DataFrame containing all predictor columns ready for ML preprocessing.
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    # Compute derived academic_pressure_score if missing
    if "academic_pressure_score" not in df.columns or df["academic_pressure_score"].isnull().any():
        df["academic_pressure_score"] = calculate_academic_pressure_score(
            academic_pressure=df["academic_pressure"],
            assignment_workload=df["assignment_workload"],
            exam_frequency=df["exam_frequency"],
            attendance_percentage=df["attendance_percentage"],
            cgpa=df["cgpa"],
            study_hours_per_day=df.get("study_hours_per_day", 4.0),
        )

    # Compute derived lifestyle_balance_score if missing
    if "lifestyle_balance_score" not in df.columns or df["lifestyle_balance_score"].isnull().any():
        df["lifestyle_balance_score"] = calculate_lifestyle_balance_score(
            sleep_hours=df["sleep_hours"],
            sleep_quality=df["sleep_quality"],
            screen_time_hours=df["screen_time_hours"],
            physical_activity_hours=df["physical_activity_hours"],
            social_interaction_hours=df["social_interaction_hours"],
            breaks_per_day=df["breaks_per_day"],
            hobbies_hours_per_week=df["hobbies_hours_per_week"],
            days_off_per_week=df["days_off_per_week"],
        )

    return df
