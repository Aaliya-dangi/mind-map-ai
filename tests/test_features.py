"""
MindMap AI - Unit Tests for Feature Engineering
Tests composite calculations, boundary conditions, monotonicity, and risk thresholding
per RMD.md §17.
"""

import pytest
import numpy as np
import pandas as pd

from features.engineering import (
    calculate_academic_pressure_score,
    calculate_lifestyle_balance_score,
    assign_risk_category,
)
from features.pipeline import enrich_features


class TestAcademicPressureScore:
    def test_minimum_boundary(self):
        """Minimum pressure inputs (relaxed student, perfect GPA & attendance) should yield low score (< 10)."""
        score = calculate_academic_pressure_score(
            academic_pressure=1,
            assignment_workload=1,
            exam_frequency=0,
            attendance_percentage=100.0,
            cgpa=10.0,
            study_hours_per_day=0.0,
        )
        assert 0.0 <= score <= 5.0

    def test_maximum_boundary(self):
        """Maximum pressure inputs (extreme pressure, failing GPA & low attendance) should yield high score (> 90)."""
        score = calculate_academic_pressure_score(
            academic_pressure=10,
            assignment_workload=10,
            exam_frequency=5,
            attendance_percentage=40.0,
            cgpa=4.0,
            study_hours_per_day=12.0,
        )
        assert 95.0 <= score <= 100.0

    def test_monotonicity_pressure_increase(self):
        """Increasing academic pressure or workload must strictly increase academic pressure score."""
        s1 = calculate_academic_pressure_score(academic_pressure=3, assignment_workload=3, exam_frequency=1, attendance_percentage=85, cgpa=8.0, study_hours_per_day=4.0)
        s2 = calculate_academic_pressure_score(academic_pressure=7, assignment_workload=3, exam_frequency=1, attendance_percentage=85, cgpa=8.0, study_hours_per_day=4.0)
        s3 = calculate_academic_pressure_score(academic_pressure=7, assignment_workload=8, exam_frequency=1, attendance_percentage=85, cgpa=8.0, study_hours_per_day=4.0)
        assert s1 < s2 < s3

    def test_inverse_relationship_attendance_cgpa(self):
        """Lower attendance and lower CGPA should produce higher academic pressure score."""
        s_high_perf = calculate_academic_pressure_score(academic_pressure=5, assignment_workload=5, exam_frequency=2, attendance_percentage=95, cgpa=9.5, study_hours_per_day=4.0)
        s_low_perf = calculate_academic_pressure_score(academic_pressure=5, assignment_workload=5, exam_frequency=2, attendance_percentage=55, cgpa=5.5, study_hours_per_day=4.0)
        assert s_low_perf > s_high_perf

    def test_vectorized_series(self):
        """Should support pandas Series vectorized inputs."""
        pressures = pd.Series([1, 5, 10])
        workloads = pd.Series([1, 5, 10])
        exams = pd.Series([0, 2, 5])
        atts = pd.Series([100.0, 80.0, 40.0])
        cgpas = pd.Series([10.0, 7.5, 4.0])
        scores = calculate_academic_pressure_score(pressures, workloads, exams, atts, cgpas)
        assert isinstance(scores, pd.Series)
        assert scores.iloc[0] < scores.iloc[1] < scores.iloc[2]


class TestLifestyleBalanceScore:
    def test_minimum_boundary(self):
        """Severe lifestyle deprivation (no sleep, high screen time, no exercise) should yield low score (< 10)."""
        score = calculate_lifestyle_balance_score(
            sleep_hours=3.0,
            sleep_quality=1,
            screen_time_hours=14.0,
            physical_activity_hours=0.0,
            social_interaction_hours=0.0,
            breaks_per_day=0,
            hobbies_hours_per_week=0.0,
            days_off_per_week=0,
        )
        assert 0.0 <= score <= 5.0

    def test_maximum_boundary(self):
        """Optimal lifestyle habits (good sleep, exercise, social time, low screen) should yield high score (> 90)."""
        score = calculate_lifestyle_balance_score(
            sleep_hours=10.0,
            sleep_quality=10,
            screen_time_hours=1.0,
            physical_activity_hours=10.0,
            social_interaction_hours=20.0,
            breaks_per_day=8,
            hobbies_hours_per_week=15.0,
            days_off_per_week=3,
        )
        assert 95.0 <= score <= 100.0

    def test_protective_factor_monotonicity(self):
        """Increasing sleep duration, physical activity, or hobbies must increase lifestyle balance score."""
        s1 = calculate_lifestyle_balance_score(sleep_hours=5.0, sleep_quality=4, screen_time_hours=8.0, physical_activity_hours=1.0, social_interaction_hours=4.0, breaks_per_day=2, hobbies_hours_per_week=2.0, days_off_per_week=1)
        s2 = calculate_lifestyle_balance_score(sleep_hours=8.0, sleep_quality=4, screen_time_hours=8.0, physical_activity_hours=1.0, social_interaction_hours=4.0, breaks_per_day=2, hobbies_hours_per_week=2.0, days_off_per_week=1)
        s3 = calculate_lifestyle_balance_score(sleep_hours=8.0, sleep_quality=4, screen_time_hours=8.0, physical_activity_hours=6.0, social_interaction_hours=4.0, breaks_per_day=2, hobbies_hours_per_week=2.0, days_off_per_week=1)
        assert s1 < s2 < s3


class TestRiskCategoryMapping:
    @pytest.mark.parametrize("score, expected_category", [
        (0.0, "Low"),
        (15.5, "Low"),
        (30.0, "Low"),
        (30.1, "Moderate"),
        (45.0, "Moderate"),
        (60.0, "Moderate"),
        (60.1, "High"),
        (75.0, "High"),
        (80.0, "High"),
        (80.1, "Very High"),
        (95.0, "Very High"),
        (100.0, "Very High"),
    ])
    def test_scalar_thresholds(self, score, expected_category):
        assert assign_risk_category(score) == expected_category

    def test_vectorized_thresholds(self):
        scores = pd.Series([10.0, 45.0, 70.0, 90.0])
        categories = assign_risk_category(scores)
        assert list(categories) == ["Low", "Moderate", "High", "Very High"]


class TestFeaturePipeline:
    def test_enrich_features_from_dict(self):
        sample = {
            "age": 20,
            "gender": "Female",
            "year_of_study": 2,
            "course": "Computer Science",
            "attendance_percentage": 85.0,
            "cgpa": 8.0,
            "study_hours_per_day": 5.0,
            "assignment_workload": 6,
            "exam_frequency": 2,
            "academic_pressure": 5,
            "sleep_hours": 7.0,
            "sleep_quality": 7,
            "screen_time_hours": 6.0,
            "physical_activity_hours": 4.0,
            "social_interaction_hours": 8.0,
            "breaks_per_day": 4,
            "hobbies_hours_per_week": 5.0,
            "days_off_per_week": 1,
        }
        df_enriched = enrich_features(sample)
        assert "academic_pressure_score" in df_enriched.columns
        assert "lifestyle_balance_score" in df_enriched.columns
        assert 0.0 <= df_enriched["academic_pressure_score"].iloc[0] <= 100.0
        assert 0.0 <= df_enriched["lifestyle_balance_score"].iloc[0] <= 100.0
