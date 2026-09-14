"""
MindMap AI - Unit Tests for What-If Simulation Scenarios
Verifies that directional input changes move model-predicted risk scores
in the expected direction per RMD.md §17.
"""

import pytest
from ml.predict import predict_burnout_risk


class TestWhatIfSimulatorScenarios:
    @pytest.fixture
    def base_student(self):
        return {
            "age": 21,
            "gender": "Female",
            "year_of_study": 3,
            "course": "Computer Science",
            "attendance_percentage": 80.0,
            "cgpa": 7.5,
            "study_hours_per_day": 6.0,
            "assignment_workload": 7,
            "exam_frequency": 3,
            "academic_pressure": 7,
            "sleep_hours": 5.0,
            "sleep_quality": 4,
            "screen_time_hours": 9.5,
            "physical_activity_hours": 1.5,
            "social_interaction_hours": 6.0,
            "breaks_per_day": 2,
            "hobbies_hours_per_week": 3.0,
            "days_off_per_week": 1,
        }

    def test_scenario_1_increasing_sleep_reduces_risk(self, base_student):
        """Increasing sleep duration from 5.0h to 8.5h should decrease risk score."""
        curr = predict_burnout_risk(base_student)
        sim_data = dict(base_student, sleep_hours=8.5, sleep_quality=8)
        sim = predict_burnout_risk(sim_data)

        assert sim["risk_score"] < curr["risk_score"]
        assert sim["lifestyle_balance_score"] > curr["lifestyle_balance_score"]

    def test_scenario_2_reducing_screen_time_reduces_risk(self, base_student):
        """Reducing daily screen time from 9.5h to 4.0h should decrease risk score."""
        curr = predict_burnout_risk(base_student)
        sim_data = dict(base_student, screen_time_hours=4.0)
        sim = predict_burnout_risk(sim_data)

        assert sim["risk_score"] <= curr["risk_score"]
        assert sim["lifestyle_balance_score"] >= curr["lifestyle_balance_score"]

    def test_scenario_3_adding_exercise_and_breaks_reduces_risk(self, base_student):
        """Adding regular exercise (from 1.5h to 6.0h) and study breaks (from 2 to 5) should reduce risk."""
        curr = predict_burnout_risk(base_student)
        sim_data = dict(base_student, physical_activity_hours=6.0, breaks_per_day=5)
        sim = predict_burnout_risk(sim_data)

        assert sim["risk_score"] < curr["risk_score"]
        assert sim["lifestyle_balance_score"] > curr["lifestyle_balance_score"]

    def test_scenario_4_increasing_study_load_increases_pressure_and_risk(self, base_student):
        """Increasing daily study hours from 6.0h to 11.0h without recovery should increase risk."""
        curr = predict_burnout_risk(base_student)
        sim_data = dict(base_student, study_hours_per_day=11.0, assignment_workload=9)
        sim = predict_burnout_risk(sim_data)

        assert sim["risk_score"] > curr["risk_score"]
        assert sim["academic_pressure_score"] > curr["academic_pressure_score"]
