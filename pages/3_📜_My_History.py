"""
MindMap AI - Page 3: My History
Personal longitudinal burnout assessment history, risk score trend line charts,
and detailed past assessment drill-downs for authenticated students.
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
from db.user_db import get_user_assessment_history

st.set_page_config(page_title="My History — MindMap AI", page_icon="📜", layout="wide")
inject_custom_css()
render_sidebar()
plotly_layout = get_plotly_layout_defaults()

user = get_current_user()
profile = get_current_profile() if user else {}
student_name = profile.get("full_name", user.get("name", "Student")) if user else "Guest"

render_header(
    title="Assessment History & Longitudinal Risk Trends",
    subtitle=f"Review your historical burnout assessments, track score trajectories over time, and inspect past model outputs · {student_name}",
    icon="📜",
)

if not is_authenticated():
    st.warning("🔒 **Authentication Required:** Please log in or create a student account to access your personal assessment history.")
    st.page_link("app.py", label="Return to Home & Log In", icon="🏠")
    st.stop()

history = get_user_assessment_history(user["user_id"])

if not history:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                    border: 1px solid #38bdf8; border-radius: 12px; padding: 2rem; text-align: center; margin-bottom: 2rem;">
            <h3 style="color:#f8fafc; margin-top:0;">📋 No Assessment History Yet</h3>
            <p style="color:#cbd5e1; max-width:600px; margin: 0.5rem auto 1.5rem auto; font-size:1rem;">
                Your assessment history and longitudinal risk trajectory will appear here after you complete your first evaluation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_🎯_My_Assessment.py", label="Start Your First Assessment", icon="🎯")
    st.stop()

# Top KPI Summary Cards for History
total_assessments = len(history)
all_scores = [h["risk_score"] for h in history]
min_score = min(all_scores)
max_score = max(all_scores)
first_date = history[0]["created_at"][:10]
latest_date = history[-1]["created_at"][:10]

h1, h2, h3, h4 = st.columns(4)
with h1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Assessments Logged</div>
            <div class="kpi-value">{total_assessments} <span style="font-size:1rem;color:#94a3b8">entries</span></div>
            <div class="kpi-subtitle">Personal evaluations</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with h2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Lowest Risk Recorded</div>
            <div class="kpi-value" style="color:#22c55e;">{min_score:.1f} <span style="font-size:1rem;color:#94a3b8">/ 100</span></div>
            <div class="kpi-subtitle">Best personal score</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with h3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Peak Risk Recorded</div>
            <div class="kpi-value" style="color:#ef4444;">{max_score:.1f} <span style="font-size:1rem;color:#94a3b8">/ 100</span></div>
            <div class="kpi-subtitle">Highest recorded strain</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with h4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Tracking Period</div>
            <div class="kpi-value" style="font-size:1.35rem; padding-top:0.4rem;">{first_date} → {latest_date}</div>
            <div class="kpi-subtitle">Evaluation timeline</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Longitudinal Trend Chart (if >= 2 assessments)
if total_assessments >= 2:
    st.markdown("### 📈 **Longitudinal Risk & Balance Trajectory**")
    st.caption("Visualizing your risk score, academic strain, and lifestyle balance across logged assessments:")

    dates = [h["created_at"][:16] for h in history]
    scores = [h["risk_score"] for h in history]
    pressures = [h["academic_pressure_score"] for h in history]
    balances = [h["lifestyle_balance_score"] for h in history]

    fig_timeline = go.Figure()

    # Background color bands for risk tiers
    fig_timeline.add_hrect(y0=0, y1=30, fillcolor="rgba(34, 197, 94, 0.08)", line_width=0, annotation_text="Low Risk Zone", annotation_position="top left")
    fig_timeline.add_hrect(y0=30, y1=60, fillcolor="rgba(234, 179, 8, 0.08)", line_width=0, annotation_text="Moderate Risk Zone", annotation_position="top left")
    fig_timeline.add_hrect(y0=60, y1=80, fillcolor="rgba(249, 115, 22, 0.08)", line_width=0, annotation_text="High Risk Zone", annotation_position="top left")
    fig_timeline.add_hrect(y0=80, y1=100, fillcolor="rgba(239, 68, 68, 0.08)", line_width=0, annotation_text="Very High Risk Zone", annotation_position="top left")

    fig_timeline.add_trace(go.Scatter(
        x=dates, y=scores, mode="lines+markers",
        name="Burnout Risk Score",
        line=dict(color="#ef4444", width=3),
        marker=dict(size=9, color="#dc2626"),
    ))
    fig_timeline.add_trace(go.Scatter(
        x=dates, y=pressures, mode="lines+markers",
        name="Academic Pressure Score",
        line=dict(color="#fb923c", width=2, dash="dot"),
        marker=dict(size=7),
    ))
    fig_timeline.add_trace(go.Scatter(
        x=dates, y=balances, mode="lines+markers",
        name="Lifestyle Balance Score",
        line=dict(color="#38bdf8", width=2, dash="dash"),
        marker=dict(size=7),
    ))

    fig_timeline.update_layout(
        template=plotly_layout["template"],
        paper_bgcolor=plotly_layout["paper_bgcolor"],
        plot_bgcolor=plotly_layout["plot_bgcolor"],
        font_color=plotly_layout["font_color"],
        height=400,
        margin=dict(t=30, b=20, l=20, r=20),
        yaxis=dict(range=[0, 105], title="Score (0–100)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_timeline)
else:
    st.info("ℹ️ **Submit more assessments to see your risk trend over time.** Longitudinal trajectory graphs automatically activate once you have 2 or more assessments logged.")

st.markdown("---")

# Summary Table of All Submissions
st.markdown("### 📋 **Assessment Submissions Log**")

history_table_data = []
for h in reversed(history):
    history_table_data.append({
        "Date & Time": h["created_at"][:19],
        "Burnout Score": f"{h['risk_score']:.1f} / 100",
        "Risk Category": h["burnout_risk"],
        "Academic Strain": f"{h['academic_pressure_score']:.1f} / 100",
        "Lifestyle Balance": f"{h['lifestyle_balance_score']:.1f} / 100",
        "Sleep Duration": f"{h['sleep_hours']:.1f} hrs",
        "Daily Study": f"{h['study_hours_per_day']:.1f} hrs",
        "Screen Time": f"{h['screen_time_hours']:.1f} hrs",
        "Breaks": f"{h['breaks_per_day']} / day",
    })

df_hist = pd.DataFrame(history_table_data)
st.dataframe(df_hist, hide_index=True)

csv_data = df_hist.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Assessment History (CSV)",
    data=csv_data,
    file_name=f"mindmap_assessment_history_{student_name.lower().replace(' ', '_')}.csv",
    mime="text/csv",
)

st.markdown("---")

# Expandable Deep Dive into Past Submissions
st.markdown("### 🔍 **Past Assessment Deep-Dives**")

for idx, h in enumerate(reversed(history)):
    date_str = h["created_at"][:19]
    with st.expander(f"📅 Assessment #{len(history) - idx}: {date_str} — {h['risk_score']:.1f}/100 ({h['burnout_risk']} Risk)", expanded=(idx == 0)):
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.markdown("##### 📝 Recorded Parameters")
            st.markdown(
                f"""
                - **Sleep:** {h['sleep_hours']} hrs/night (Quality: {h['sleep_quality']}/10)
                - **Study Schedule:** {h['study_hours_per_day']} hrs/day ({h['breaks_per_day']} breaks/day)
                - **Workload & Pressure:** Workload {h['assignment_workload']}/10 · Pressure {h['academic_pressure']}/10
                - **Screen Time:** {h['screen_time_hours']} hrs/day
                - **Exercise & Hobbies:** Exercise {h['physical_activity_hours']} hrs/wk · Hobbies {h['hobbies_hours_per_week']} hrs/wk · Days Off: {h['days_off_per_week']}
                """
            )

            st.markdown("##### 📈 Model Probabilities")
            for cat, prob in h.get("probabilities", {}).items():
                st.write(f"- **{cat}:** {prob*100:.1f}%")

        with col_d2:
            st.markdown("##### 🔍 Contributing Factors")
            factors = h.get("contributing_factors", [])
            if factors:
                for f in factors:
                    icon = "🔺" if f.get("direction") == "elevates_risk" else "🛡️"
                    st.markdown(f"{icon} **{f.get('label', '')}:** {f.get('explanation', '')}")
            else:
                st.write("Baseline median habits.")

            st.markdown("##### 💡 Saved Recommendations")
            recs = h.get("recommendations", [])
            for r in recs:
                st.markdown(f"- **{r.get('title', '')}:** {r.get('suggested_action', '')}")

render_disclaimer()
