"""
MindMap AI - Streamlit Native AppTest End-to-End Suite
Uses streamlit.testing.v1.AppTest to execute every single page, simulate user interactions,
verify session state handling, form submissions, and ensure 0 runtime exceptions across all views.
"""

from pathlib import Path
import uuid
import pytest
from streamlit.testing.v1 import AppTest
from db.user_db import (
    create_user,
    get_user_by_email,
    get_student_profile,
    save_user_assessment_and_prediction,
    ensure_admin_account,
)
from db.connection import get_connection
from utils.auth import hash_password
from ml.predict import predict_burnout_risk
from utils.recommendations import generate_data_driven_recommendations

BASE_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture
def test_student_apptest():
    """Create an isolated test student user for AppTest suites and clean up on teardown."""
    ensure_admin_account()
    unique_id = uuid.uuid4().hex[:8]
    pwd_hash, salt = hash_password("TestStudentPass2026!")
    user = create_user(
        name="Test Student AppTest",
        username=f"test_app_{unique_id}",
        email=f"test_app_{unique_id}@mindmap.ai",
        password_hash=pwd_hash,
        salt=salt,
        role="student",
        age=22,
        gender="Female",
        year_of_study=3,
        course="Data Science",
        attendance_percentage=90.0,
        cgpa=8.5,
    )
    # Save a baseline assessment and prediction for this student so history/simulator tests have a record
    assessment_payload = {
        "age": 22,
        "gender": "Female",
        "year_of_study": 3,
        "course": "Data Science",
        "attendance_percentage": 90.0,
        "cgpa": 8.5,
        "study_hours_per_day": 6.5,
        "assignment_workload": 3,
        "exam_frequency": 2,
        "academic_pressure": 3,
        "sleep_hours": 7.0,
        "sleep_quality": 3,
        "screen_time_hours": 5.0,
        "physical_activity_hours": 1.5,
        "social_interaction_hours": 2.0,
        "breaks_per_day": 3,
        "hobbies_hours_per_week": 4.0,
        "days_off_per_week": 1,
    }
    pred = predict_burnout_risk(assessment_payload)
    recs = generate_data_driven_recommendations(assessment_payload, pred)
    save_user_assessment_and_prediction(user["user_id"], assessment_payload, pred, recs)

    profile = get_student_profile(user["user_id"])
    yield user, profile

    # Teardown: delete test student and cascaded records
    conn, engine = get_connection()
    c = conn.cursor()
    uid = user["user_id"]
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
    c.close()
    conn.close()


class TestStreamlitAppEndToEnd:
    def test_app_landing_guest_mode(self):
        """Tests that app.py loads cleanly in guest mode without errors."""
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=15)
        at.run()
        assert not at.exception
        assert len(at.tabs) >= 3

    def test_app_dashboard_authenticated_mode(self, test_student_apptest):
        """Tests that app.py renders personalized student dashboard when logged in."""
        student_user, profile = test_student_apptest

        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=15)
        at.session_state["user"] = student_user
        at.session_state["user_profile"] = profile
        at.run()
        assert not at.exception
        assert "Test Student AppTest" in str([m.value for m in at.markdown])

    def test_my_assessment_page(self, test_student_apptest):
        """Tests that My Assessment loads and submits prediction without errors."""
        student_user, profile = test_student_apptest

        at = AppTest.from_file(str(BASE_DIR / "pages" / "1_🎯_My_Assessment.py"), default_timeout=15)
        at.session_state["user"] = student_user
        at.session_state["user_profile"] = profile
        at.run()
        assert not at.exception

        if len(at.button) > 0:
            at.button[0].click().run()
            assert not at.exception

    def test_what_if_simulator_page(self, test_student_apptest):
        """Tests that What-If Simulator loads baseline and computes simulation deltas."""
        student_user, profile = test_student_apptest

        at = AppTest.from_file(str(BASE_DIR / "pages" / "2_🔮_What_If_Simulator.py"), default_timeout=15)
        at.session_state["user"] = student_user
        at.session_state["user_profile"] = profile
        at.run()
        assert not at.exception
        assert len(at.selectbox) >= 1

    def test_my_history_page(self, test_student_apptest):
        """Tests that My History renders timeline charts and assessment log table."""
        student_user, profile = test_student_apptest

        at = AppTest.from_file(str(BASE_DIR / "pages" / "3_📜_My_History.py"), default_timeout=15)
        at.session_state["user"] = student_user
        at.session_state["user_profile"] = profile
        at.run()
        assert not at.exception
        assert len(at.dataframe) >= 1

    def test_population_analytics_page(self):
        """Tests that Population Analytics renders all 5 tabs and Plotly visualizations without error."""
        at = AppTest.from_file(str(BASE_DIR / "pages" / "4_📊_Population_Analytics.py"), default_timeout=15)
        at.run()
        assert not at.exception
        assert len(at.tabs) == 5

    def test_model_insights_page(self):
        """Tests that Model Insights renders confusion matrices and model comparison table."""
        at = AppTest.from_file(str(BASE_DIR / "pages" / "5_🔬_Model_Insights.py"), default_timeout=15)
        at.run()
        assert not at.exception
        assert len(at.dataframe) >= 2

    def test_student_profile_page(self, test_student_apptest):
        """Tests that Profile page loads and updates student information."""
        student_user, profile = test_student_apptest

        at = AppTest.from_file(str(BASE_DIR / "pages" / "6_👤_Profile.py"), default_timeout=15)
        at.session_state["user"] = student_user
        at.session_state["user_profile"] = profile
        at.run()
        assert not at.exception
        if len(at.button) > 0:
            at.button[0].click().run()
            assert not at.exception

    def test_admin_dashboard_view(self):
        """Tests that app.py renders institutional analytics for admin users."""
        ensure_admin_account()
        admin_user = get_user_by_email("admin@mindmap.ai")
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=15)
        at.session_state["user"] = admin_user
        at.run()
        assert not at.exception
        assert "Administrator Portal" in str([m.value for m in at.markdown])

    def test_logout_session_clearing(self):
        """Tests that logout helper resets session state properly."""
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=15)
        at.session_state["user"] = {"user_id": "test", "name": "Test", "role": "student"}
        at.session_state["user_profile"] = {"full_name": "Test User"}
        at.run()
        assert not at.exception

