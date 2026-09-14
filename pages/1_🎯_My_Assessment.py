"""
MindMap AI - Page 1: My Assessment
Interactive multi-section student burnout assessment form with live ML prediction,
risk score, category badge, explainable contributing factors, data-driven recommendations,
and automatic historical persistence for authenticated students.
"""

import sys
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

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
    get_risk_badge_html,
    get_plotly_layout_defaults,
)
from utils.auth import is_authenticated, get_current_user, get_current_profile
from db.user_db import save_user_assessment_and_prediction
from ml.predict import predict_burnout_risk
from utils.recommendations import generate_data_driven_recommendations
from features.definitions import CATEGORICAL_VOCABULARIES

st.set_page_config(page_title="My Assessment — MindMap AI", page_icon="🎯", layout="wide")
inject_custom_css()
render_sidebar()
plotly_layout = get_plotly_layout_defaults()

# Get profile if logged in
user = get_current_user()
profile = get_current_profile() if user else {}

student_name = profile.get("full_name", user.get("name", "Student")) if user else "Guest"
default_age = profile.get("age", 21) if user else 21
default_gender = profile.get("gender", "Female") if user else "Female"
default_year = profile.get("year_of_study", 2) if user else 2
default_course = profile.get("course", "Computer Science") if user else "Computer Science"
default_cgpa = profile.get("cgpa", 8.0) if user else 8.0
default_att = profile.get("attendance_percentage", 85.0) if user else 85.0

render_header(
    title="Personal Burnout Risk Assessment",
    subtitle=f"Complete your workload and lifestyle assessment to generate a live, ML-calibrated risk evaluation · {student_name}",
    icon="🎯",
)

if not is_authenticated():
    st.info("💡 **Tip:** You are currently in Guest Mode. You can complete the assessment to test the model, but logging in will save your assessment history and timeline trends!")

with st.form("burnout_assessment_form"):
    st.markdown("### 📝 **1. Academic Context & Performance Baseline**")
    st.caption("Pre-populated from your student profile (you can adjust if needed):")

    c_p1, c_p2, c_p3 = st.columns(3)
    with c_p1:
        age_in = st.number_input("Age", min_value=17, max_value=35, value=int(default_age))
        gender_in = st.selectbox("Gender", CATEGORICAL_VOCABULARIES["gender"], index=CATEGORICAL_VOCABULARIES["gender"].index(default_gender) if default_gender in CATEGORICAL_VOCABULARIES["gender"] else 0)
    with c_p2:
        year_in = st.selectbox("Year of Study", [1, 2, 3, 4], index=int(default_year) - 1 if default_year in [1, 2, 3, 4] else 1)
        course_in = st.selectbox("Course / Major", CATEGORICAL_VOCABULARIES["course"], index=CATEGORICAL_VOCABULARIES["course"].index(default_course) if default_course in CATEGORICAL_VOCABULARIES["course"] else 3)
    with c_p3:
        cgpa_in = st.slider("Current CGPA (Scale of 10)", min_value=4.0, max_value=10.0, value=float(default_cgpa), step=0.1)
        attendance_in = st.slider("Attendance Rate (%)", min_value=40.0, max_value=100.0, value=float(default_att), step=1.0)

    st.markdown("---")
    st.markdown("### 📚 **2. Academic Workload & Pressure Signals**")
    st.caption("Enter your current academic study schedule and perceived workload intensity:")

    c_w1, c_w2 = st.columns(2)
    with c_w1:
        study_hours = st.slider("Daily Study Hours", min_value=0.5, max_value=12.0, value=6.0, step=0.5, help="Average focused study hours per day")
        assignment_workload = st.slider("Assignment Workload Intensity (1–10)", min_value=1, max_value=10, value=6, help="1 = Very Light, 10 = Severe / Compounding")
    with c_w2:
        exam_frequency = st.slider("Monthly Exams Count", min_value=0, max_value=5, value=2, help="Number of exams or major evaluations per month")
        academic_pressure = st.slider("Self-Reported Academic Pressure (1–10)", min_value=1, max_value=10, value=6, help="Subjective feeling of academic pressure")

    st.markdown("---")
    st.markdown("### 🌿 **3. Lifestyle Habits & Recovery Buffers**")
    st.caption("Enter your daily sleep patterns, screen exposure, and restorative habits:")

    c_l1, c_l2, c_l3 = st.columns(3)
    with c_l1:
        sleep_hours = st.slider("Nightly Sleep Duration (hours)", min_value=3.0, max_value=10.0, value=6.2, step=0.2)
        sleep_quality = st.slider("Sleep Quality (1–10)", min_value=1, max_value=10, value=6, help="1 = Restless/Insomnia, 10 = Deep & Fully Refreshing")
        screen_time = st.slider("Daily Screen Time (hours)", min_value=1.0, max_value=14.0, value=8.0, step=0.5)
    with c_l2:
        physical_activity = st.slider("Physical Activity (hours/week)", min_value=0.0, max_value=10.0, value=3.0, step=0.5)
        social_interaction = st.slider("Social Interaction (hours/week)", min_value=0.0, max_value=20.0, value=8.0, step=0.5)
        breaks_per_day = st.slider("Study Rest Breaks per Day", min_value=0, max_value=8, value=3)
    with c_l3:
        hobbies = st.slider("Hobbies & Creative Leisure (hours/week)", min_value=0.0, max_value=15.0, value=4.0, step=0.5)
        days_off = st.slider("Complete Days Off per Week", min_value=0, max_value=3, value=1)

    submitted = st.form_submit_button("🚀 Check My Burnout Risk", width="stretch", type="primary")

if submitted:
    student_payload = {
        "age": age_in,
        "gender": gender_in,
        "year_of_study": year_in,
        "course": course_in,
        "attendance_percentage": attendance_in,
        "cgpa": cgpa_in,
        "study_hours_per_day": study_hours,
        "assignment_workload": assignment_workload,
        "exam_frequency": exam_frequency,
        "academic_pressure": academic_pressure,
        "sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality,
        "screen_time_hours": screen_time,
        "physical_activity_hours": physical_activity,
        "social_interaction_hours": social_interaction,
        "breaks_per_day": breaks_per_day,
        "hobbies_hours_per_week": hobbies,
        "days_off_per_week": days_off,
    }

    # Pass through trained ML inference pipeline
    prediction = predict_burnout_risk(student_payload)
    recommendations = generate_data_driven_recommendations(student_payload, prediction)

    # Persist to DB if authenticated
    if is_authenticated():
        save_user_assessment_and_prediction(user["user_id"], student_payload, prediction, recommendations)
        st.success("✅ Assessment successfully recorded to your personal student history!")

    st.markdown("---")
    st.markdown("### 📊 **Assessment Results & Statistical Profile**")

    r_col1, r_col2 = st.columns([1, 1.2])

    with r_col1:
        badge_html = get_risk_badge_html(prediction["predicted_category"])
        st.markdown(
            f"""
            <div class="kpi-card" style="text-align:center; padding: 1.5rem; margin-bottom: 1rem;">
                <div class="kpi-title">Predicted Burnout Risk Score</div>
                <div class="kpi-value" style="font-size: 3rem; margin: 0.3rem 0;">
                    {prediction['risk_score']} <span style="font-size: 1.1rem; color: #94a3b8">/ 100</span>
                </div>
                <div>{badge_html}</div>
                <div style="margin-top: 1rem; font-size: 0.875rem; color: #94a3b8;">
                    Academic Strain: <strong>{prediction['academic_pressure_score']}/100</strong> | 
                    Lifestyle Balance: <strong>{prediction['lifestyle_balance_score']}/100</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Gauge Chart
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prediction["risk_score"],
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                    "bar": {"color": "#38bdf8"},
                    "steps": [
                        {"range": [0, 30], "color": "rgba(34, 197, 94, 0.25)"},
                        {"range": [30, 60], "color": "rgba(234, 179, 8, 0.25)"},
                        {"range": [60, 80], "color": "rgba(249, 115, 22, 0.25)"},
                        {"range": [80, 100], "color": "rgba(239, 68, 68, 0.25)"},
                    ],
                },
            )
        )
        fig_gauge.update_layout(
            template=plotly_layout["template"],
            paper_bgcolor=plotly_layout["paper_bgcolor"],
            plot_bgcolor=plotly_layout["plot_bgcolor"],
            font_color=plotly_layout["font_color"],
            height=200,
            margin=dict(t=5, b=5, l=15, r=15),
        )
        st.plotly_chart(fig_gauge)

        # Multi-class Probabilities
        st.markdown("##### 📈 Model Class Probabilities")
        prob_df = pd.DataFrame(
            [{"Category": k, "Probability": v * 100} for k, v in prediction["probabilities"].items()]
        )
        fig_prob = px.bar(
            prob_df,
            x="Probability",
            y="Category",
            orientation="h",
            color="Category",
            color_discrete_map={"Low": "#22c55e", "Moderate": "#eab308", "High": "#f97316", "Very High": "#ef4444"},
            text=prob_df["Probability"].apply(lambda v: f"{v:.1f}%"),
        )
        fig_prob.update_traces(textposition="outside")
        fig_prob.update_layout(
            template=plotly_layout["template"],
            paper_bgcolor=plotly_layout["paper_bgcolor"],
            plot_bgcolor=plotly_layout["plot_bgcolor"],
            font_color=plotly_layout["font_color"],
            height=180,
            margin=dict(t=5, b=5, l=10, r=10),
            showlegend=False,
            xaxis=dict(range=[0, 105]),
        )
        st.plotly_chart(fig_prob)

    with r_col2:
        st.markdown("#### 🔍 **Explainable Contributing Factors**")
        st.caption("Specific drivers and protective buffers relative to the student baseline:")

        factors = prediction["contributing_factors"]
        if factors:
            for factor in factors:
                direction_class = "elevates" if factor["direction"] == "elevates_risk" else "lowers"
                icon = "🔺" if factor["direction"] == "elevates_risk" else "🛡️"
                st.markdown(
                    f"""
                    <div class="factor-card {direction_class}">
                        <div style="font-weight:600; font-size:0.95rem; margin-bottom:0.25rem;">
                            {icon} {factor['label']}
                        </div>
                        <div style="font-size:0.875rem;">
                            {factor['explanation']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("Your inputs match the student cohort median benchmarks closely.")

        # Data-Driven Recommendations Section
        st.markdown("---")
        st.markdown("#### 💡 **Actionable Pacing & Habit Recommendations**")
        st.caption("Data-driven observations derived from cohort statistical benchmarks:")

        for rec in recommendations:
            priority_color = "#ef4444" if rec["priority"] == "High Impact" else "#38bdf8" if rec["priority"] == "Moderate Impact" else "#22c55e"
            st.markdown(
                f"""
                <div class="factor-card" style="border-left: 4px solid {priority_color}; margin-bottom:0.75rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;">
                        <span style="font-weight:600; font-size:0.95rem;">
                            {rec['icon']} {rec['title']}
                        </span>
                        <span style="font-size:0.75rem; padding:0.15rem 0.5rem; border-radius:4px; background:rgba(255,255,255,0.1); color:{priority_color};">
                            {rec['priority']}
                        </span>
                    </div>
                    <p style="font-size:0.85rem; color:#94a3b8; margin:0.25rem 0 0.4rem 0;">
                        {rec['observation']}
                    </p>
                    <p style="font-size:0.85rem; color:#38bdf8; margin:0;">
                        <strong>Action:</strong> {rec['suggested_action']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

render_disclaimer()
