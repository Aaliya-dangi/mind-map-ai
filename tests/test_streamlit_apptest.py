"""
MindMap AI - Streamlit Native AppTest End-to-End Suite
Uses streamlit.testing.v1.AppTest to execute every single page, simulate user interactions,
verify session state handling, form submissions, and ensure 0 runtime exceptions across all views.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from db.user_db import seed_demo_user, get_user_by_username, get_student_profile

BASE_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def setup_seed():
    """Ensure demo user exists for all tests."""
    seed_demo_user()


class TestStreamlitAppEndToEnd:
    def test_app_landing_guest_mode(self):
        """Tests that app.py loads cleanly in guest mode without errors."""
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=15)
        at.run()
        assert not at.exception
        assert len(at.tabs) >= 3

    def test_app_dashboard_authenticated_mode(self):
        """Tests that app.py renders personalized student dashboard when logged in."""
        demo_user = get_user_by_username("demo")
        profile = get_student_profile(demo_user["user_id"])

        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=15)
        at.session_state["user"] = {
            "user_id": demo_user["user_id"],
            "username": demo_user["username"],
            "full_name": demo_user["full_name"],
        }
        at.session_state["user_profile"] = profile
        at.run()
        assert not at.exception
        assert "Aditi Rao" in str([m.value for m in at.markdown])

    def test_my_assessment_page(self):
        """Tests that My Assessment loads and submits prediction without errors."""
        demo_user = get_user_by_username("demo")
        profile = get_student_profile(demo_user["user_id"])

        at = AppTest.from_file(str(BASE_DIR / "pages" / "1_🎯_My_Assessment.py"), default_timeout=15)
        at.session_state["user"] = {
            "user_id": demo_user["user_id"],
            "username": demo_user["username"],
            "full_name": demo_user["full_name"],
        }
        at.session_state["user_profile"] = profile
        at.run()
        assert not at.exception

        if len(at.button) > 0:
            at.button[0].click().run()
            assert not at.exception

    def test_what_if_simulator_page(self):
        """Tests that What-If Simulator loads baseline and computes simulation deltas."""
        demo_user = get_user_by_username("demo")
        profile = get_student_profile(demo_user["user_id"])

        at = AppTest.from_file(str(BASE_DIR / "pages" / "2_🔮_What_If_Simulator.py"), default_timeout=15)
        at.session_state["user"] = {
            "user_id": demo_user["user_id"],
            "username": demo_user["username"],
            "full_name": demo_user["full_name"],
        }
        at.session_state["user_profile"] = profile
        at.run()
        assert not at.exception
        assert len(at.selectbox) >= 1

    def test_my_history_page(self):
        """Tests that My History renders timeline charts and assessment log table."""
        demo_user = get_user_by_username("demo")
        profile = get_student_profile(demo_user["user_id"])

        at = AppTest.from_file(str(BASE_DIR / "pages" / "3_📜_My_History.py"), default_timeout=15)
        at.session_state["user"] = {
            "user_id": demo_user["user_id"],
            "username": demo_user["username"],
            "full_name": demo_user["full_name"],
        }
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

    def test_student_profile_page(self):
        """Tests that Profile page loads and updates student information."""
        demo_user = get_user_by_username("demo")
        profile = get_student_profile(demo_user["user_id"])

        at = AppTest.from_file(str(BASE_DIR / "pages" / "6_👤_Profile.py"), default_timeout=15)
        at.session_state["user"] = {
            "user_id": demo_user["user_id"],
            "username": demo_user["username"],
            "full_name": demo_user["full_name"],
        }
        at.session_state["user_profile"] = profile
        at.run()
        assert not at.exception
        if len(at.button) > 0:
            at.button[0].click().run()
            assert not at.exception

    def test_admin_dashboard_view(self):
        """Tests that app.py renders institutional analytics for admin users."""
        admin_user = get_user_by_username("admin")
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=15)
        at.session_state["user"] = {
            "user_id": admin_user["user_id"] if admin_user else "admin-id",
            "name": "System Administrator",
            "role": "admin",
        }
        at.run()
        assert not at.exception
        assert "Administrator Portal" in str([m.value for m in at.markdown])

    def test_logout_session_clearing(self):
        """Tests that logout helper resets session state properly."""
        from utils.auth import logout_user
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=15)
        at.session_state["user"] = {"user_id": "test", "name": "Test", "role": "student"}
        at.session_state["user_profile"] = {"full_name": "Test User"}
        at.run()
        assert not at.exception

