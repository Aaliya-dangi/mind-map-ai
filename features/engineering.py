"""
MindMap AI - Feature Engineering
Implements composite metric formulas for academic_pressure_score and lifestyle_balance_score,
as well as risk category mapping per design.md §18 and §23.
"""

import numpy as np
import pandas as pd
from typing import Union


def calculate_academic_pressure_score(
    academic_pressure: Union[float, int, np.ndarray, pd.Series],
    assignment_workload: Union[float, int, np.ndarray, pd.Series],
    exam_frequency: Union[float, int, np.ndarray, pd.Series],
    attendance_percentage: Union[float, int, np.ndarray, pd.Series],
    cgpa: Union[float, int, np.ndarray, pd.Series],
    study_hours_per_day: Union[float, int, np.ndarray, pd.Series] = 4.0,
) -> Union[float, np.ndarray, pd.Series]:
    """
    Computes normalized Academic Pressure Score (0-100).
    Higher score indicates higher composite academic strain.

    Components:
    - academic_pressure (scale 1-10): self-reported pressure (weight: 0.30)
    - assignment_workload (scale 1-10): workload volume (weight: 0.25)
    - exam_frequency (scale 0-5): monthly exams (weight: 0.15)
    - study_hours_per_day (scale 0-12): daily study hours (weight: 0.10)
    - inverse attendance (40-100%): lower attendance increases risk (weight: 0.10)
    - inverse cgpa (4.0-10.0): academic struggle/pressure indicator (weight: 0.10)
    """
    # Normalize inputs to 0-100 scale
    norm_pressure = (np.clip(academic_pressure, 1, 10) - 1.0) / 9.0 * 100.0
    norm_workload = (np.clip(assignment_workload, 1, 10) - 1.0) / 9.0 * 100.0
    norm_exams = np.clip(exam_frequency, 0, 5) / 5.0 * 100.0
    norm_study = np.clip(study_hours_per_day, 0, 12) / 12.0 * 100.0

    # Inverse metrics (lower attendance / lower cgpa -> higher pressure signal)
    norm_inv_att = (100.0 - np.clip(attendance_percentage, 40.0, 100.0)) / 60.0 * 100.0
    norm_inv_cgpa = (10.0 - np.clip(cgpa, 4.0, 10.0)) / 6.0 * 100.0

    composite = (
        0.30 * norm_pressure
        + 0.25 * norm_workload
        + 0.15 * norm_exams
        + 0.10 * norm_study
        + 0.10 * norm_inv_att
        + 0.10 * norm_inv_cgpa
    )

    score = np.clip(composite, 0.0, 100.0)
    if isinstance(academic_pressure, (int, float)):
        return round(float(score), 2)
    elif isinstance(academic_pressure, pd.Series):
        return score.round(2)
    return np.round(score, 2)


def calculate_lifestyle_balance_score(
    sleep_hours: Union[float, int, np.ndarray, pd.Series],
    sleep_quality: Union[float, int, np.ndarray, pd.Series],
    screen_time_hours: Union[float, int, np.ndarray, pd.Series],
    physical_activity_hours: Union[float, int, np.ndarray, pd.Series],
    social_interaction_hours: Union[float, int, np.ndarray, pd.Series],
    breaks_per_day: Union[float, int, np.ndarray, pd.Series],
    hobbies_hours_per_week: Union[float, int, np.ndarray, pd.Series],
    days_off_per_week: Union[float, int, np.ndarray, pd.Series],
) -> Union[float, np.ndarray, pd.Series]:
    """
    Computes normalized Lifestyle Balance Score (0-100).
    Higher score indicates healthier, more balanced lifestyle habits.

    Components:
    - sleep_hours (scale 3-10): duration (weight: 0.20)
    - sleep_quality (scale 1-10): restfulness (weight: 0.15)
    - inverse screen_time (scale 1-14): lower excessive screen time is healthier (weight: 0.15)
    - physical_activity_hours (scale 0-10/wk): exercise (weight: 0.15)
    - social_interaction_hours (scale 0-20/wk): social support (weight: 0.10)
    - breaks_per_day (scale 0-8): rest intervals (weight: 0.10)
    - hobbies_hours_per_week (scale 0-15/wk): recreational relaxation (weight: 0.10)
    - days_off_per_week (scale 0-3): recovery days (weight: 0.05)
    """
    norm_sleep_h = (np.clip(sleep_hours, 3.0, 10.0) - 3.0) / 7.0 * 100.0
    norm_sleep_q = (np.clip(sleep_quality, 1, 10) - 1.0) / 9.0 * 100.0
    norm_screen_inv = (14.0 - np.clip(screen_time_hours, 1.0, 14.0)) / 13.0 * 100.0
    norm_phys = np.clip(physical_activity_hours, 0.0, 10.0) / 10.0 * 100.0
    norm_social = np.clip(social_interaction_hours, 0.0, 20.0) / 20.0 * 100.0
    norm_breaks = np.clip(breaks_per_day, 0, 8) / 8.0 * 100.0
    norm_hobbies = np.clip(hobbies_hours_per_week, 0.0, 15.0) / 15.0 * 100.0
    norm_days_off = np.clip(days_off_per_week, 0, 3) / 3.0 * 100.0

    composite = (
        0.20 * norm_sleep_h
        + 0.15 * norm_sleep_q
        + 0.15 * norm_screen_inv
        + 0.15 * norm_phys
        + 0.10 * norm_social
        + 0.10 * norm_breaks
        + 0.10 * norm_hobbies
        + 0.05 * norm_days_off
    )

    score = np.clip(composite, 0.0, 100.0)
    if isinstance(sleep_hours, (int, float)):
        return round(float(score), 2)
    elif isinstance(sleep_hours, pd.Series):
        return score.round(2)
    return np.round(score, 2)


def assign_risk_category(score: Union[float, int, np.ndarray, pd.Series]) -> Union[str, np.ndarray, pd.Series]:
    """
    Maps burnout risk score (0-100) to standard categorical risk buckets:
    - 0 to 30: Low
    - 30.1 to 60: Moderate
    - 60.1 to 80: High
    - 80.1 to 100: Very High
    """
    if isinstance(score, (int, float)):
        if score <= 30.0:
            return "Low"
        elif score <= 60.0:
            return "Moderate"
        elif score <= 80.0:
            return "High"
        else:
            return "Very High"
    
    # Vectorized logic for pandas Series / numpy array
    conditions = [
        score <= 30.0,
        (score > 30.0) & (score <= 60.0),
        (score > 60.0) & (score <= 80.0),
        score > 80.0,
    ]
    choices = ["Low", "Moderate", "High", "Very High"]
    
    if isinstance(score, pd.Series):
        return pd.Series(np.select(conditions, choices, default="Moderate"), index=score.index)
    return np.select(conditions, choices, default="Moderate")
