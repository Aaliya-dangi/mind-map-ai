"""
MindMap AI - Unit & Integration Tests for Authentication, Student Profile, and User Data Flow
Validates secure password hashing, user registration, authentication, profile updates,
assessment persistence, and history retrieval.
"""

import uuid
import pytest
import pandas as pd
from utils.auth import hash_password, verify_password, signup_user, login_user
from db.user_db import (
    create_user,
    get_user_by_username,
    get_user_by_email,
    get_student_profile,
    update_student_profile,
    save_user_assessment_and_prediction,
    get_user_assessment_history,
    get_latest_user_prediction,
    ensure_admin_account,
)
from db.connection import get_connection
from ml.predict import predict_burnout_risk
from utils.recommendations import generate_data_driven_recommendations


class TestAuthenticationSecurity:
    def test_password_hashing_and_verification(self):
        """Hashes should verify correctly and never match plaintext or different salts."""
        raw_pwd = "SecureStudentPass2026!"
        h1, s1 = hash_password(raw_pwd)
        h2, s2 = hash_password(raw_pwd)

        # Unique salts should produce different hashes
        assert s1 != s2
        assert h1 != h2
        assert verify_password(raw_pwd, h1, s1) is True
        assert verify_password(raw_pwd, h2, s2) is True
        assert verify_password("WrongPassword123", h1, s1) is False

    def test_duplicate_user_creation_blocked(self):
        """Duplicate usernames must be rejected."""
        unique_uname = f"test_dup_{uuid.uuid4().hex[:8]}"
        pwd_hash, salt = hash_password("pass123456")
        
        u1 = create_user(username=unique_uname, password_hash=pwd_hash, salt=salt, full_name="Student One")
        assert u1 is not None
        
        # Second creation attempt should fail
        u2 = create_user(username=unique_uname, password_hash=pwd_hash, salt=salt, full_name="Student Two")
        assert u2 is None

        # Clean up
        conn, engine = get_connection()
        c = conn.cursor()
        if engine == "sqlite":
            c.execute("DELETE FROM student_profiles WHERE user_id = ?", (u1["user_id"],))
            c.execute("DELETE FROM users WHERE user_id = ?", (u1["user_id"],))
        else:
            c.execute("DELETE FROM student_profiles WHERE user_id = %s", (u1["user_id"],))
            c.execute("DELETE FROM users WHERE user_id = %s", (u1["user_id"],))
        conn.commit()
        conn.close()


class TestStudentProfileAndWorkflow:
    @pytest.fixture
    def test_student(self):
        uname = f"student_{uuid.uuid4().hex[:8]}"
        pwd = "ValidPassword123"
        pwd_hash, salt = hash_password(pwd)
        user_info = create_user(
            username=uname,
            password_hash=pwd_hash,
            salt=salt,
            full_name="Alex Mercer",
            age=22,
            gender="Male",
            year_of_study=3,
            course="Computer Science",
            attendance_percentage=88.5,
            cgpa=8.5,
        )
        yield user_info, uname, pwd

        # Teardown / Cleanup
        if user_info:
            conn, engine = get_connection()
            c = conn.cursor()
            uid = user_info["user_id"]
            if engine == "sqlite":
                c.execute("DELETE FROM predictions WHERE user_id = ?", (uid,))
                c.execute("DELETE FROM assessments WHERE user_id = ?", (uid,))
                c.execute("DELETE FROM student_profiles WHERE user_id = ?", (uid,))
                c.execute("DELETE FROM users WHERE user_id = ?", (uid,))
            else:
                c.execute("DELETE FROM predictions WHERE user_id = %s", (uid,))
                c.execute("DELETE FROM assessments WHERE user_id = %s", (uid,))
                c.execute("DELETE FROM student_profiles WHERE user_id = %s", (uid,))
                c.execute("DELETE FROM users WHERE user_id = %s", (uid,))
            conn.commit()
            conn.close()

    def test_profile_fetch_and_update(self, test_student):
        """Student profile should accurately fetch and update."""
        user_info, uname, _ = test_student
        user_id = user_info["user_id"]

        profile = get_student_profile(user_id)
        assert profile is not None
        assert profile["full_name"] == "Alex Mercer"
        assert profile["course"] == "Computer Science"
        assert profile["cgpa"] == 8.5

        # Update profile
        updated = update_student_profile(
            user_id=user_id,
            full_name="Alex Mercer Jr.",
            age=23,
            gender="Male",
            year_of_study=4,
            course="Computer Science",
            attendance_percentage=91.0,
            cgpa=8.9,
        )
        assert updated is True

        new_profile = get_student_profile(user_id)
        assert new_profile["full_name"] == "Alex Mercer Jr."
        assert new_profile["year_of_study"] == 4
        assert new_profile["cgpa"] == 8.9

    def test_assessment_saving_and_history_retrieval(self, test_student):
        """Assessments and ML predictions must persist and retrieve in historical order."""
        user_info, _, _ = test_student
        user_id = user_info["user_id"]

        assessment_1 = {
            "age": 22,
            "gender": "Male",
            "year_of_study": 3,
            "course": "Computer Science",
            "attendance_percentage": 88.5,
            "cgpa": 8.5,
            "study_hours_per_day": 8.0,
            "assignment_workload": 8,
            "exam_frequency": 3,
            "academic_pressure": 8,
            "sleep_hours": 5.2,
            "sleep_quality": 4,
            "screen_time_hours": 9.0,
            "physical_activity_hours": 1.0,
            "social_interaction_hours": 4.0,
            "breaks_per_day": 2,
            "hobbies_hours_per_week": 2.0,
            "days_off_per_week": 0,
        }

        # Run model inference
        pred_1 = predict_burnout_risk(assessment_1)
        recs_1 = generate_data_driven_recommendations(assessment_1, pred_1)
        
        # Save to DB
        aid_1 = save_user_assessment_and_prediction(user_id, assessment_1, pred_1, recs_1)
        assert aid_1 is not None

        # Check latest prediction
        latest = get_latest_user_prediction(user_id)
        assert latest is not None
        assert latest["assessment_id"] == aid_1
        assert latest["risk_score"] == pred_1["risk_score"]
        assert latest["burnout_risk"] == pred_1["predicted_category"]

        # Log a second improved assessment
        assessment_2 = dict(assessment_1, sleep_hours=8.0, sleep_quality=8, study_hours_per_day=5.0, days_off_per_week=2)
        pred_2 = predict_burnout_risk(assessment_2)
        recs_2 = generate_data_driven_recommendations(assessment_2, pred_2)
        aid_2 = save_user_assessment_and_prediction(user_id, assessment_2, pred_2, recs_2)

        # Check history contains 2 items
        history = get_user_assessment_history(user_id)
        assert len(history) == 2
        assert history[0]["assessment_id"] == aid_1
        assert history[1]["assessment_id"] == aid_2
        assert history[1]["risk_score"] < history[0]["risk_score"]

    def test_admin_account_initialization(self):
        """ensure_admin_account function should ensure the admin user exists."""
        ensure_admin_account()
        admin = get_user_by_email("admin@mindmap.ai")
        assert admin is not None
        assert admin["role"] == "admin"

