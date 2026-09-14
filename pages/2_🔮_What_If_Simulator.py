"""
MindMap AI - Page 2: What-If Simulator
Interactive counterfactual simulation allowing students to experiment with
habit modifications (sleep, study breaks, screen time) against their actual
saved baseline assessment and observe live ML-calibrated risk deltas.
"""

import sys
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
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
from db.user_db import get_latest_user_prediction, get_user_assessment_history
from ml.predict import predict_burnout_risk

st.set_page_config(page_title="What-If Simulator — MindMap AI", page_icon="🔮", layout="wide")
inject_custom_css()
render_sidebar()
plotly_layout = get_plotly_layout_defaults()

render_header(
    title="What-If Burnout Simulator",
    subtitle="Simulate lifestyle and study habit adjustments against your real assessment baseline to observe live model risk deltas",
    icon="🔮",
)

# Authenticated student check & fetch latest baseline
user = get_current_user()
profile = get_current_profile() if user else {}
history = get_user_assessment_history(user["user_id"]) if user else []

# =====================================================================
# 1. TOP CARD: CURRENT BURNOUT RISK BASELINE
# =====================================================================
if not is_authenticated():
    st.warning("🔒 **Authentication Required:** Please log in as a student to load your personalized baseline for simulation.")
    st.info("👉 Return to the **Home** page to log in or use the 1-Click Quick Demo.")
    st.stop()

if not history:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                    border: 1px solid #38bdf8; border-radius: 12px; padding: 2rem; text-align: center; margin-bottom: 2rem;">
            <h3 style="color:#f8fafc; margin-top:0;">📋 No Assessment Available Yet</h3>
            <p style="color:#cbd5e1; max-width:600px; margin: 0.5rem auto 1.5rem auto; font-size:1rem;">
                The What-If Simulator requires an initial burnout assessment as your active baseline. 
                Complete a 2-minute assessment to unlock personalized habit simulations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("🚀 Complete an Assessment", width="stretch", type="primary"):
        st.info("👉 Please select **My Assessment** in the sidebar navigation to complete your assessment.")
    st.stop()

# Select from student's actual assessments
history_options = {
    f"Assessment on {h['created_at'][:16]} (Risk: {h['risk_score']:.1f}/100 · {h['burnout_risk']} Risk)": h
    for h in reversed(history)
}
selected_label = st.selectbox(
    "🎯 **Select Student Baseline Assessment for Simulation:**",
    list(history_options.keys()),
    index=0,
)
latest_pred = history_options[selected_label]

# Display Current Burnout Risk Baseline Card
badge_curr = get_risk_badge_html(latest_pred["burnout_risk"])
last_date = latest_pred["created_at"][:10]

st.markdown("### 📌 **Current Burnout Risk Baseline**")
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f"""
        <div class="kpi-card" style="text-align:center;">
            <div class="kpi-title">Current Risk Score</div>
            <div class="kpi-value">{latest_pred['risk_score']:.1f} <span style="font-size:1rem;color:#94a3b8">/ 100</span></div>
            <div>{badge_curr}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""
        <div class="kpi-card" style="text-align:center;">
            <div class="kpi-title">Academic Strain Index</div>
            <div class="kpi-value" style="color:#ef4444">{latest_pred['academic_pressure_score']:.1f} <span style="font-size:1rem;color:#94a3b8">/ 100</span></div>
            <div class="kpi-subtitle">Workload & exam stress</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""
        <div class="kpi-card" style="text-align:center;">
            <div class="kpi-title">Lifestyle Balance Index</div>
            <div class="kpi-value" style="color:#38bdf8">{latest_pred['lifestyle_balance_score']:.1f} <span style="font-size:1rem;color:#94a3b8">/ 100</span></div>
            <div class="kpi-subtitle">Rest & recovery buffer</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""
        <div class="kpi-card" style="text-align:center;">
            <div class="kpi-title">Baseline Assessment Date</div>
            <div class="kpi-value" style="font-size:1.4rem; padding-top:0.4rem;">{last_date}</div>
            <div class="kpi-subtitle">Active student reference</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# =====================================================================
# 2. INTERACTIVE SIMULATION INPUTS (Initialized to Current Baseline)
# =====================================================================
st.markdown("### 🎛️ **Modify Habit Variables for Simulation**")
st.caption("Adjust the sliders below to explore how micro-adjustments in your routine alter model-predicted risk:")

# Session state initialization for slider values if reset
if "sim_reset_trigger" not in st.session_state:
    st.session_state["sim_reset_trigger"] = 0

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("##### 🛌 Sleep & Digital Wellness Adjustments")
    sim_sleep = st.slider(
        "Simulated Sleep Duration (hours/night):",
        min_value=3.0, max_value=10.0,
        value=float(latest_pred["sleep_hours"]),
        step=0.2,
        key=f"sim_sleep_{st.session_state['sim_reset_trigger']}",
        help=f"Current baseline: {latest_pred['sleep_hours']} hrs",
    )
    sim_sq = st.slider(
        "Simulated Sleep Quality (1–10):",
        min_value=1, max_value=10,
        value=int(latest_pred["sleep_quality"]),
        step=1,
        key=f"sim_sq_{st.session_state['sim_reset_trigger']}",
        help=f"Current baseline: {latest_pred['sleep_quality']} / 10",
    )
    sim_screen = st.slider(
        "Simulated Screen Time (hours/day):",
        min_value=1.0, max_value=14.0,
        value=float(latest_pred["screen_time_hours"]),
        step=0.5,
        key=f"sim_screen_{st.session_state['sim_reset_trigger']}",
        help=f"Current baseline: {latest_pred['screen_time_hours']} hrs",
    )

with col_right:
    st.markdown("##### 📚 Study Schedule & Restorative Habits")
    sim_study = st.slider(
        "Simulated Study Hours (hours/day):",
        min_value=0.5, max_value=12.0,
        value=float(latest_pred["study_hours_per_day"]),
        step=0.5,
        key=f"sim_study_{st.session_state['sim_reset_trigger']}",
        help=f"Current baseline: {latest_pred['study_hours_per_day']} hrs",
    )
    sim_breaks = st.slider(
        "Simulated Study Rest Breaks (per day):",
        min_value=0, max_value=8,
        value=int(latest_pred["breaks_per_day"]),
        step=1,
        key=f"sim_breaks_{st.session_state['sim_reset_trigger']}",
        help=f"Current baseline: {latest_pred['breaks_per_day']} breaks/day",
    )
    sim_phys = st.slider(
        "Simulated Physical Activity (hours/week):",
        min_value=0.0, max_value=10.0,
        value=float(latest_pred["physical_activity_hours"]),
        step=0.5,
        key=f"sim_phys_{st.session_state['sim_reset_trigger']}",
        help=f"Current baseline: {latest_pred['physical_activity_hours']} hrs/wk",
    )
    sim_days_off = st.slider(
        "Simulated Complete Days Off (per week):",
        min_value=0, max_value=3,
        value=int(latest_pred["days_off_per_week"]),
        step=1,
        key=f"sim_days_off_{st.session_state['sim_reset_trigger']}",
        help=f"Current baseline: {latest_pred['days_off_per_week']} days/wk",
    )

# =====================================================================
# 3. ACTION BUTTONS: [Run Simulation], [Compare Results], [Reset]
# =====================================================================
st.markdown("###")
btn_col1, btn_col2, btn_col3 = st.columns(3)

with btn_col1:
    run_sim = st.button("🚀 Run Simulation", type="primary", width="stretch")

with btn_col2:
    compare_btn = st.button("📊 Compare Results", width="stretch")

with btn_col3:
    reset_btn = st.button("🔄 Reset to Current Values", width="stretch")

if reset_btn:
    st.session_state["sim_reset_trigger"] += 1
    if "sim_results" in st.session_state:
        del st.session_state["sim_results"]
    st.success("Values reset to your latest assessment baseline.")
    st.rerun()

# Execute real ML simulation on demand
if run_sim or compare_btn:
    # Build complete modified payload
    simulated_payload = {
        "age": profile.get("age", 21),
        "gender": profile.get("gender", "Female"),
        "year_of_study": profile.get("year_of_study", 2),
        "course": profile.get("course", "Computer Science"),
        "attendance_percentage": float(latest_pred["attendance_percentage"]),
        "cgpa": float(latest_pred["cgpa"]),
        "assignment_workload": int(latest_pred["assignment_workload"]),
        "exam_frequency": int(latest_pred["exam_frequency"]),
        "academic_pressure": int(latest_pred["academic_pressure"]),
        "study_hours_per_day": float(sim_study),
        "sleep_hours": float(sim_sleep),
        "sleep_quality": int(sim_sq),
        "screen_time_hours": float(sim_screen),
        "physical_activity_hours": float(sim_phys),
        "social_interaction_hours": float(latest_pred["social_interaction_hours"]),
        "breaks_per_day": int(sim_breaks),
        "hobbies_hours_per_week": float(latest_pred["hobbies_hours_per_week"]),
        "days_off_per_week": int(sim_days_off),
    }

    # Pass through the exact trained ML inference pipeline
    sim_pred = predict_burnout_risk(simulated_payload)

    # Store simulation outcome in session state (exploratory, zero DB write)
    st.session_state["sim_results"] = {
        "sim_payload": simulated_payload,
        "sim_pred": sim_pred,
    }

# =====================================================================
# 4. SIMULATION COMPARISON & INTERPRETATION RESULTS
# =====================================================================
if "sim_results" in st.session_state:
    sim_data = st.session_state["sim_results"]
    sim_pred = sim_data["sim_pred"]

    curr_score = float(latest_pred["risk_score"])
    sim_score = float(sim_pred["risk_score"])
    delta = round(sim_score - curr_score, 1)

    curr_pressure = float(latest_pred["academic_pressure_score"])
    sim_pressure = float(sim_pred["academic_pressure_score"])
    pressure_delta = round(sim_pressure - curr_pressure, 1)

    curr_balance = float(latest_pred["lifestyle_balance_score"])
    sim_balance = float(sim_pred["lifestyle_balance_score"])
    balance_delta = round(sim_balance - curr_balance, 1)

    st.markdown("---")
    st.markdown("### 📊 **Simulation Comparison & Outcome**")

    # 3 Summary Comparison Cards: Current vs Simulated vs Delta
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="kpi-card" style="text-align:center; padding:1.25rem;">
                <div class="kpi-title">Current Risk</div>
                <div class="kpi-value" style="color:#ef4444;">{curr_score:.1f} <span style="font-size:1rem;color:#94a3b8">/ 100</span></div>
                <div>{get_risk_badge_html(latest_pred['burnout_risk'])}</div>
                <div style="margin-top:0.6rem; font-size:0.8rem; color:#94a3b8;">
                    Strain: <strong>{curr_pressure:.1f}</strong> | Buffer: <strong>{curr_balance:.1f}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="kpi-card" style="text-align:center; padding:1.25rem;">
                <div class="kpi-title">Simulated Risk</div>
                <div class="kpi-value" style="color:#38bdf8;">{sim_score:.1f} <span style="font-size:1rem;color:#94a3b8">/ 100</span></div>
                <div>{get_risk_badge_html(sim_pred['predicted_category'])}</div>
                <div style="margin-top:0.6rem; font-size:0.8rem; color:#94a3b8;">
                    Strain: <strong>{sim_pressure:.1f}</strong> | Buffer: <strong>{sim_balance:.1f}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        delta_color = "#22c55e" if delta <= 0 else "#ef4444"
        delta_sign = f"-{abs(delta)}" if delta < 0 else f"+{delta}" if delta > 0 else "0.0"
        st.markdown(
            f"""
            <div class="kpi-card" style="text-align:center; padding:1.25rem;">
                <div class="kpi-title">Net Risk Change</div>
                <div class="kpi-value" style="color:{delta_color}; font-size:2.4rem;">
                    {delta_sign} <span style="font-size:1rem;">pts</span>
                </div>
                <div style="color:#94a3b8; font-size:0.8rem; margin-top:0.25rem;">
                    Lifestyle Buffer: <strong style="color:#22c55e">{balance_delta:+0.1f} pts</strong><br>
                    Academic Strain: <strong style="color:{'#22c55e' if pressure_delta <= 0 else '#ef4444'}">{pressure_delta:+0.1f} pts</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("###")

    # Clear Verbal Interpretation
    if delta < -3.0:
        st.success(f"🎉 **Positive Outcome:** Your simulated risk **decreased by {abs(delta):.1f} points** under these changed inputs.")
    elif delta > 3.0:
        st.warning(f"⚠️ **Elevated Risk:** Your simulated risk **increased by {delta:.1f} points** under these changed inputs.")
    else:
        st.info("ℹ️ **Stable Trajectory:** The simulated modifications maintain your burnout risk score within the same bracket.")

    # Side-by-Side Bar Chart Comparison
    fig_comp = go.Figure(
        data=[
            go.Bar(
                name="Current Risk Baseline",
                x=["Burnout Risk Score", "Academic Strain Index", "Lifestyle Balance Index"],
                y=[curr_score, curr_pressure, curr_balance],
                marker_color="#ef4444",
                text=[f"{curr_score:.1f}", f"{curr_pressure:.1f}", f"{curr_balance:.1f}"],
                textposition="outside",
            ),
            go.Bar(
                name="Simulated Outcome",
                x=["Burnout Risk Score", "Academic Strain Index", "Lifestyle Balance Index"],
                y=[sim_score, sim_pressure, sim_balance],
                marker_color="#38bdf8",
                text=[f"{sim_score:.1f}", f"{sim_pressure:.1f}", f"{sim_balance:.1f}"],
                textposition="outside",
            ),
        ]
    )
    fig_comp.update_layout(
        barmode="group",
        template=plotly_layout["template"],
        paper_bgcolor=plotly_layout["paper_bgcolor"],
        plot_bgcolor=plotly_layout["plot_bgcolor"],
        font_color=plotly_layout["font_color"],
        yaxis=dict(range=[0, 110], title="Index Score (0–100)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=30, b=20, l=20, r=20),
        height=360,
    )
    st.plotly_chart(fig_comp)

    # Changed Variables Summary Table
    st.markdown("##### 📝 Summary of Modified Variables")
    diff_rows = [
        {"Variable": "Sleep Duration", "Current Value": f"{latest_pred['sleep_hours']:.1f} hrs/night", "Simulated Value": f"{sim_sleep:.1f} hrs/night", "Difference": f"{sim_sleep - float(latest_pred['sleep_hours']):+0.1f} hrs"},
        {"Variable": "Sleep Quality", "Current Value": f"{latest_pred['sleep_quality']} / 10", "Simulated Value": f"{sim_sq} / 10", "Difference": f"{sim_sq - int(latest_pred['sleep_quality']):+d}"},
        {"Variable": "Study Hours", "Current Value": f"{latest_pred['study_hours_per_day']:.1f} hrs/day", "Simulated Value": f"{sim_study:.1f} hrs/day", "Difference": f"{sim_study - float(latest_pred['study_hours_per_day']):+0.1f} hrs"},
        {"Variable": "Screen Time", "Current Value": f"{latest_pred['screen_time_hours']:.1f} hrs/day", "Simulated Value": f"{sim_screen:.1f} hrs/day", "Difference": f"{sim_screen - float(latest_pred['screen_time_hours']):+0.1f} hrs"},
        {"Variable": "Study Breaks", "Current Value": f"{latest_pred['breaks_per_day']} / day", "Simulated Value": f"{sim_breaks} / day", "Difference": f"{sim_breaks - int(latest_pred['breaks_per_day']):+d}"},
        {"Variable": "Physical Activity", "Current Value": f"{latest_pred['physical_activity_hours']:.1f} hrs/wk", "Simulated Value": f"{sim_phys:.1f} hrs/wk", "Difference": f"{sim_phys - float(latest_pred['physical_activity_hours']):+0.1f} hrs"},
        {"Variable": "Days Off", "Current Value": f"{latest_pred['days_off_per_week']} days/wk", "Simulated Value": f"{sim_days_off} days/wk", "Difference": f"{sim_days_off - int(latest_pred['days_off_per_week']):+d}"},
    ]
    df_diff = pd.DataFrame(diff_rows)
    st.dataframe(df_diff, hide_index=True)

else:
    st.info("💡 Adjust any variables above and click **🚀 Run Simulation** to compute the live model prediction.")

render_disclaimer(config.SIMULATOR_DISCLAIMER)
