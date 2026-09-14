"""
MindMap AI - Unit Tests for Live Prediction & Explainability
Tests model loading, probability extraction, scoring, and explainability output.
"""

import pytest
from ml.predict import predict_burnout_risk, load_artifacts


class TestLivePredictionPipeline:
    @classmethod
    def setup_class(cls):
        """Ensure artifacts are loaded."""
        load_artifacts()

    def test_high_stress_student_prediction(self):
        """A student with severe sleep deprivation and high workload should receive elevated risk."""
        high_stress_student = {
            "age": 22,
            "gender": "Male",
            "year_of_study": 4,
            "course": "Medicine",
            "attendance_percentage": 65.0,
            "cgpa": 6.5,
            "study_hours_per_day": 9.5,
            "assignment_workload": 9,
            "exam_frequency": 4,
            "academic_pressure": 9,
            "sleep_hours": 4.0,
            "sleep_quality": 3,
            "screen_time_hours": 11.0,
            "physical_activity_hours": 0.5,
            "social_interaction_hours": 2.0,
            "breaks_per_day": 1,
            "hobbies_hours_per_week": 1.0,
            "days_off_per_week": 0,
        }
        res = predict_burnout_risk(high_stress_student)
        assert res["predicted_category"] in ["High", "Very High"]
        assert res["risk_score"] >= 60.0
        assert res["academic_pressure_score"] > 60.0
        assert res["lifestyle_balance_score"] < 40.0
        assert len(res["contributing_factors"]) > 0
        assert any(f["direction"] == "elevates_risk" for f in res["contributing_factors"])
        assert "disclaimer" in res

    def test_balanced_lifestyle_student_prediction(self):
        """A student with healthy sleep, exercise, and manageable workload should receive low/moderate risk."""
        balanced_student = {
            "age": 19,
            "gender": "Female",
            "year_of_study": 1,
            "course": "Arts",
            "attendance_percentage": 92.0,
            "cgpa": 8.5,
            "study_hours_per_day": 3.0,
            "assignment_workload": 3,
            "exam_frequency": 1,
            "academic_pressure": 2,
            "sleep_hours": 8.5,
            "sleep_quality": 9,
            "screen_time_hours": 4.0,
            "physical_activity_hours": 6.0,
            "social_interaction_hours": 14.0,
            "breaks_per_day": 6,
            "hobbies_hours_per_week": 10.0,
            "days_off_per_week": 2,
        }
        res = predict_burnout_risk(balanced_student)
        assert res["predicted_category"] in ["Low", "Moderate"]
        assert res["risk_score"] <= 45.0
        assert res["academic_pressure_score"] < 35.0
        assert res["lifestyle_balance_score"] > 65.0
        assert len(res["contributing_factors"]) > 0
        assert any(f["direction"] == "lowers_risk" for f in res["contributing_factors"])

    def test_probability_distribution(self):
        """Class probabilities should sum to approximately 1.0."""
        sample = {
            "age": 20,
            "gender": "Other",
            "year_of_study": 2,
            "course": "Computer Science",
            "attendance_percentage": 80.0,
            "cgpa": 7.5,
            "study_hours_per_day": 5.0,
            "assignment_workload": 6,
            "exam_frequency": 2,
            "academic_pressure": 5,
            "sleep_hours": 6.5,
            "sleep_quality": 6,
            "screen_time_hours": 7.5,
            "physical_activity_hours": 4.0,
            "social_interaction_hours": 8.0,
            "breaks_per_day": 3,
            "hobbies_hours_per_week": 5.0,
            "days_off_per_week": 1,
        }
        res = predict_burnout_risk(sample)
        prob_sum = sum(res["probabilities"].values())
        assert pytest.approx(prob_sum, rel=1e-2) == 1.0
