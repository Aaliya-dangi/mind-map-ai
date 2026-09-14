"""
MindMap AI - Feature Definitions & Metadata
Central repository of feature schemas, types, units, realistic bounds, and display labels.
"""

from typing import Dict, Any, List

# Raw Categorical Features
CATEGORICAL_FEATURES = ["gender", "course"]

# Categorical Feature Vocabularies
CATEGORICAL_VOCABULARIES = {
    "gender": ["Female", "Male", "Other"],
    "course": ["Arts", "Business", "Commerce", "Computer Science", "Design", "Engineering", "Medicine"],
}

# Raw Numerical Features
RAW_NUMERICAL_FEATURES = [
    "age",
    "year_of_study",
    "attendance_percentage",
    "cgpa",
    "study_hours_per_day",
    "assignment_workload",
    "exam_frequency",
    "academic_pressure",
    "sleep_hours",
    "sleep_quality",
    "screen_time_hours",
    "physical_activity_hours",
    "social_interaction_hours",
    "breaks_per_day",
    "hobbies_hours_per_week",
    "days_off_per_week",
]

# Derived Engineered Features
DERIVED_NUMERICAL_FEATURES = [
    "academic_pressure_score",
    "lifestyle_balance_score",
]

# All ML Predictor Features (Raw + Derived)
ALL_PREDICTOR_FEATURES = (
    RAW_NUMERICAL_FEATURES + DERIVED_NUMERICAL_FEATURES + CATEGORICAL_FEATURES
)

# Feature metadata (for forms, sliders, and visualizations)
FEATURE_METADATA: Dict[str, Dict[str, Any]] = {
    "age": {"label": "Age", "type": "int", "min": 18, "max": 28, "default": 21, "unit": "years"},
    "gender": {"label": "Gender", "type": "categorical", "options": ["Female", "Male", "Other"], "default": "Female"},
    "year_of_study": {"label": "Year of Study", "type": "int", "min": 1, "max": 4, "default": 2, "unit": "year"},
    "course": {
        "label": "Course / Major",
        "type": "categorical",
        "options": ["Arts", "Business", "Commerce", "Computer Science", "Design", "Engineering", "Medicine"],
        "default": "Computer Science",
    },
    "attendance_percentage": {"label": "Attendance", "type": "float", "min": 40.0, "max": 100.0, "default": 82.0, "unit": "%"},
    "cgpa": {"label": "CGPA", "type": "float", "min": 4.0, "max": 10.0, "default": 7.5, "unit": "GPA (scale 10)"},
    "study_hours_per_day": {"label": "Daily Study Hours", "type": "float", "min": 0.5, "max": 12.0, "default": 5.0, "unit": "hrs/day"},
    "assignment_workload": {"label": "Assignment Workload", "type": "int", "min": 1, "max": 10, "default": 6, "unit": "scale (1-10)"},
    "exam_frequency": {"label": "Monthly Exams", "type": "int", "min": 0, "max": 5, "default": 2, "unit": "exams/mo"},
    "academic_pressure": {"label": "Academic Pressure", "type": "int", "min": 1, "max": 10, "default": 5, "unit": "scale (1-10)"},
    "sleep_hours": {"label": "Sleep Duration", "type": "float", "min": 3.0, "max": 10.0, "default": 6.8, "unit": "hrs/night"},
    "sleep_quality": {"label": "Sleep Quality", "type": "int", "min": 1, "max": 10, "default": 7, "unit": "scale (1-10)"},
    "screen_time_hours": {"label": "Screen Time", "type": "float", "min": 1.0, "max": 14.0, "default": 7.5, "unit": "hrs/day"},
    "physical_activity_hours": {"label": "Physical Activity", "type": "float", "min": 0.0, "max": 10.0, "default": 4.5, "unit": "hrs/wk"},
    "social_interaction_hours": {"label": "Social Interaction", "type": "float", "min": 0.0, "max": 20.0, "default": 10.0, "unit": "hrs/wk"},
    "breaks_per_day": {"label": "Study Breaks", "type": "int", "min": 0, "max": 8, "default": 4, "unit": "breaks/day"},
    "hobbies_hours_per_week": {"label": "Hobbies / Leisure", "type": "float", "min": 0.0, "max": 15.0, "default": 6.5, "unit": "hrs/wk"},
    "days_off_per_week": {"label": "Days Off", "type": "int", "min": 0, "max": 3, "default": 1, "unit": "days/wk"},
    "academic_pressure_score": {"label": "Academic Pressure Score", "type": "float", "min": 0.0, "max": 100.0, "unit": "index (0-100)"},
    "lifestyle_balance_score": {"label": "Lifestyle Balance Score", "type": "float", "min": 0.0, "max": 100.0, "unit": "index (0-100)"},
}
