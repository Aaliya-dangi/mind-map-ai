"""
MindMap AI - Unit Tests for Data-Driven Recommendations
Verifies that recommendation logic correctly triggers based on specific
student vulnerability vectors and non-medical phrasing.
"""

import pytest
from utils.recommendations import generate_data_driven_recommendations
from ml.predict import predict_burnout_risk


class TestDataDrivenRecommendations:
    def test_sleep_deprivation_triggers_sleep_recommendation(self):
        """Student with < 6.0h sleep must receive High Impact Sleep recommendation."""
        student = {
            "age": 21, "gender": "Male", "year_of_study": 3, "course": "Engineering",
            "attendance_percentage": 80.0, "cgpa": 7.5, "study_hours_per_day": 6.0,
            "assignment_workload": 6, "exam_frequency": 2, "academic_pressure": 6,
            "sleep_hours": 4.5, "sleep_quality": 3, "screen_time_hours": 7.0,
            "physical_activity_hours": 4.0, "social_interaction_hours": 8.0,
            "breaks_per_day": 3, "hobbies_hours_per_week": 4.0, "days_off_per_week": 1,
        }
        pred = predict_burnout_risk(student)
        recs = generate_data_driven_recommendations(student, pred)

        categories = [r["category"] for r in recs]
        assert "Sleep & Recovery" in categories
        sleep_rec = next(r for r in recs if r["category"] == "Sleep & Recovery")
        assert sleep_rec["priority"] == "High Impact"
        assert "7.0" in sleep_rec["observation"]

    def test_screen_time_triggers_digital_wellness_recommendation(self):
        """Student with excessive screen time (>= 8.5h) must receive Digital Wellness recommendation."""
        student = {
            "age": 20, "gender": "Female", "year_of_study": 2, "course": "Design",
            "attendance_percentage": 85.0, "cgpa": 8.0, "study_hours_per_day": 5.0,
            "assignment_workload": 6, "exam_frequency": 2, "academic_pressure": 5,
            "sleep_hours": 7.0, "sleep_quality": 7, "screen_time_hours": 11.5,
            "physical_activity_hours": 4.0, "social_interaction_hours": 8.0,
            "breaks_per_day": 4, "hobbies_hours_per_week": 5.0, "days_off_per_week": 1,
        }
        pred = predict_burnout_risk(student)
        recs = generate_data_driven_recommendations(student, pred)

        categories = [r["category"] for r in recs]
        assert "Digital Wellness" in categories

    def test_resilient_student_receives_maintenance_reinforcement(self):
        """Student with balanced metrics and low risk should receive Maintenance reinforcement."""
        student = {
            "age": 19, "gender": "Female", "year_of_study": 1, "course": "Arts",
            "attendance_percentage": 95.0, "cgpa": 9.0, "study_hours_per_day": 3.0,
            "assignment_workload": 2, "exam_frequency": 1, "academic_pressure": 2,
            "sleep_hours": 8.5, "sleep_quality": 9, "screen_time_hours": 4.0,
            "physical_activity_hours": 6.0, "social_interaction_hours": 14.0,
            "breaks_per_day": 5, "hobbies_hours_per_week": 9.0, "days_off_per_week": 2,
        }
        pred = predict_burnout_risk(student)
        recs = generate_data_driven_recommendations(student, pred)

        categories = [r["category"] for r in recs]
        assert "Maintenance" in categories
        maint_rec = next(r for r in recs if r["category"] == "Maintenance")
        assert maint_rec["priority"] == "Positive Reinforcement"
