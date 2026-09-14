"""
MindMap AI - Page 6: Student Profile
View and manage registered student academic background, enrolled department,
year of study, CGPA, and attendance baseline metrics.
"""

import sys
from pathlib import Path
import streamlit as st

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from utils.ui_components import (
    inject_custom_css,
    render_header,
    render_sidebar,
    render_disclaimer,
    get_current_theme,
)
from utils.auth import (
    is_authenticated,
    get_current_user,
    get_current_profile,
    refresh_user_profile,
)
from db.user_db import update_student_profile, get_student_profile
from features.definitions import CATEGORICAL_VOCABULARIES

st.set_page_config(page_title="Student Profile — MindMap AI", page_icon="👤", layout="wide")
inject_custom_css()
render_sidebar()

render_header(
    title="Student Profile & Academic Baseline",
    subtitle="Manage your personal academic baseline, enrolled department, year of study, and performance parameters",
    icon="👤",
)

if not is_authenticated():
    st.warning("🔒 **Authentication Required:** Please log in or create a student account to access and edit your personal profile.")
    st.info("👉 Head back to the **Home** page to log in or use the 1-click Quick Demo.")
    st.stop()

user = get_current_user()
profile = get_current_profile() or {}

user_id = user.get("user_id", "")
current_name = profile.get("full_name", user.get("name", "Student"))
current_email = profile.get("email", user.get("email", "student@mindmap.ai"))
current_age = profile.get("age", 21)
current_gender = profile.get("gender", "Female")
current_year = profile.get("year_of_study", 2)
current_course = profile.get("course", "Computer Science")
current_cgpa = profile.get("cgpa", 8.0)
current_attendance = profile.get("attendance_percentage", 85.0)

theme = get_current_theme()
is_dark = theme == "dark"
card_bg = "#1e293b" if is_dark else "#ffffff"
card_border = "#38bdf8" if is_dark else "#0284c7"
title_color = "#f8fafc" if is_dark else "#0f172a"
accent_color = "#38bdf8" if is_dark else "#0284c7"
sub_color = "#94a3b8" if is_dark else "#475569"

# Profile Overview Card
st.markdown(
    f"""
    <div style="background:{card_bg}; border:1px solid {card_border}; border-radius:10px; padding: 1.5rem; margin-bottom: 2rem; box-shadow:0 2px 4px rgba(0,0,0,0.06);">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
            <div>
                <h2 style="margin:0; color:{title_color};">🎓 {current_name}</h2>
                <div style="color:{accent_color}; font-weight:600; font-size:1rem; margin-top:0.25rem;">
                    {current_course} · Year {current_year}
                </div>
                <div style="color:{sub_color}; font-size:0.875rem; margin-top:0.35rem;">
                    📧 {current_email} · Student ID: <code style="color:{sub_color};">{user_id[:8]}...</code>
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:1.6rem; font-weight:700; color:#22c55e;">CGPA: {current_cgpa:.1f}</div>
                <div style="color:{sub_color}; font-size:0.85rem;">Attendance: {current_attendance:.0f}%</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Edit Profile Form
st.markdown("### ✏️ **Edit Student Academic Baseline**")
st.caption("Update your registered details below. These values automatically pre-populate your burnout assessments:")

with st.form("edit_student_profile_form"):
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### 👤 Personal Information")
        new_name = st.text_input("Full Name", value=current_name)
        new_age = st.number_input("Age", min_value=17, max_value=35, value=int(current_age))
        
        gender_index = CATEGORICAL_VOCABULARIES["gender"].index(current_gender) if current_gender in CATEGORICAL_VOCABULARIES["gender"] else 0
        new_gender = st.selectbox("Gender", CATEGORICAL_VOCABULARIES["gender"], index=gender_index)

    with c2:
        st.markdown("##### 📚 Academic Program")
        course_index = CATEGORICAL_VOCABULARIES["course"].index(current_course) if current_course in CATEGORICAL_VOCABULARIES["course"] else 3
        new_course = st.selectbox("Course / Department", CATEGORICAL_VOCABULARIES["course"], index=course_index)

        year_options = [1, 2, 3, 4]
        year_index = year_options.index(current_year) if current_year in year_options else 1
        new_year = st.selectbox("Year of Study", year_options, index=year_index)

        p_sub1, p_sub2 = st.columns(2)
        with p_sub1:
            new_cgpa = st.number_input("Current CGPA (Scale 10)", min_value=4.0, max_value=10.0, value=float(current_cgpa), step=0.1)
        with p_sub2:
            new_attendance = st.number_input("Attendance Rate (%)", min_value=40.0, max_value=100.0, value=float(current_attendance), step=1.0)

    st.markdown("###")
    save_btn = st.form_submit_button("💾 Save Profile Changes", width="stretch", type="primary")

    if save_btn:
        if not new_name.strip():
            st.error("Full name cannot be empty.")
        else:
            success = update_student_profile(
                user_id=user_id,
                full_name=new_name.strip(),
                name=new_name.strip(),
                age=int(new_age),
                gender=new_gender,
                year_of_study=int(new_year),
                course=new_course,
                attendance_percentage=float(new_attendance),
                cgpa=float(new_cgpa),
            )
            if success:
                # Refresh session state
                updated_p = get_student_profile(user_id)
                st.session_state["user_profile"] = updated_p
                if "user" in st.session_state and st.session_state["user"]:
                    st.session_state["user"]["name"] = new_name.strip()
                    st.session_state["user"]["full_name"] = new_name.strip()
                st.success("🎉 Profile updated successfully!")
                st.rerun()
            else:
                st.error("Failed to update profile. Please try again.")

render_disclaimer()
