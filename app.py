"""
MindMap AI - Main Application Router & Home Dashboard
Acts as the central router providing:
1. Guest View: Interactive landing & multi-role authentication portal.
2. Student View: Personalized student wellness dashboard with live ML predictions,
   contributing factors, pacing recommendations, and risk trends.
3. Administrator View: Live institutional analytics, dynamic SWOT analysis,
   cohort risk monitoring, and student records management.
"""

import sys
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Set up page config FIRST
st.set_page_config(
    page_title="MindMap AI — Home",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add project root to path for imports
BASE_DIR = Path(__file__).resolve().parent
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
    get_current_theme,
)
from utils.auth import (
    is_authenticated,
    is_admin,
    is_student,
    get_current_user,
    get_current_profile,
    render_auth_section,
    logout_user,
)
from db.user_db import (
    get_latest_user_prediction,
    get_user_assessment_history,
    get_admin_dashboard_kpis,
    get_admin_risk_distribution,
    get_admin_swot_analysis,
    get_admin_student_records,
    seed_default_accounts,
)

# Initialize accounts, styles & sidebar
seed_default_accounts()
inject_custom_css()
render_sidebar()
plotly_layout = get_plotly_layout_defaults()

# =====================================================================
# 1. GUEST LANDING VIEW (Unauthenticated)
# =====================================================================
if not is_authenticated():
    render_header(
        title="Home",
        subtitle="Personalized Student Burnout Intelligence & Institutional Risk Analytics Platform",
        icon="🧠",
    )

    st.markdown(
        """
        > **"Understand the Pattern. Predict the Risk. Take Back Control."**
        
        Welcome to **MindMap AI** — an end-to-end data science platform that analyzes academic workloads, 
        sleep habits, and recovery signals to predict student burnout risk and provide explainable, actionable recommendations.
        """
    )

    # Render Role-Based Authentication Tabs (Student Login, Student Signup, Admin Login, Quick 1-Click Demo)
    render_auth_section()

    st.markdown("---")

    # Platform Feature Highlights
    st.markdown("### 🗺️ **Platform Capabilities & Architecture**")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(
            """
            #### 🎓 Student Portal
            - **Personalized Risk Scoring:** Calibrated multi-class ML burnout predictions.
            - **Explainable Factors:** Key stress drivers (🔺) and protective buffers (🛡️).
            - **What-If Simulator:** Interactive simulations to forecast risk reductions.
            - **Longitudinal History:** Track your personal wellness trajectory over time.
            """
        )
    with p2:
        st.markdown(
            """
            #### 🛡️ Institutional Admin Portal
            - **Dynamic Aggregation:** Real-time KPIs computed strictly from live database records.
            - **Data-Driven SWOT:** Automated institutional Strengths, Weaknesses, Opportunities, and Threats.
            - **Cohort Demographics:** Multi-dimensional drill-downs by department, year, and risk tier.
            - **Student Assessment Records:** Searchable, privacy-conscious records management.
            """
        )
    with p3:
        st.markdown(
            """
            #### 🔬 Machine Learning Pipeline
            - **Trained ML Models:** Evaluated ensembles (Random Forest, Gradient Boosting, Logistic Regression).
            - **Dual-Engine DB:** SQLite embedded engine with MySQL support.
            - **Zero Plaintext:** PBKDF2-HMAC salted authentication.
            """
        )

    render_disclaimer()

# =====================================================================
# 2. ADMINISTRATOR DASHBOARD (Role: Admin)
# =====================================================================
elif is_admin():
    user = get_current_user()
    admin_name = user.get("name", "Administrator")

    render_header(
        title=f"Administrator Portal · {admin_name}",
        subtitle="Live Institutional Burnout Monitoring & Cohort Intelligence Overview",
        icon="🛡️",
    )

    # Fetch live dynamic metrics strictly from actual database records
    kpis = get_admin_dashboard_kpis()

    # 4 Primary Institutional KPI Summary Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Registered Students</div>
                <div class="kpi-value">{kpis['total_students']} <span style="font-size:1rem;color:#94a3b8">students</span></div>
                <div class="kpi-subtitle">Registered student accounts in DB</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Assessments Logged</div>
                <div class="kpi-value">{kpis['total_assessments']} <span style="font-size:1rem;color:#94a3b8">submissions</span></div>
                <div class="kpi-subtitle">Actual student evaluations</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        burnout_val = f"{kpis['avg_burnout_score']} / 100" if kpis['total_assessments'] > 0 else "0.0 / 100"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Cohort Avg Burnout Risk</div>
                <div class="kpi-value">{burnout_val}</div>
                <div class="kpi-subtitle">Mean predicted score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">High-Risk Proportion</div>
                <div class="kpi-value" style="color:#ef4444">{kpis['high_risk_percentage']}%</div>
                <div class="kpi-subtitle">{kpis['high_risk_count']} in High / Very High</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("###")

    # Dynamic Risk Distribution & Institutional Averages
    df_risk = get_admin_risk_distribution()
    color_map = {"Low": "#22c55e", "Moderate": "#eab308", "High": "#f97316", "Very High": "#ef4444"}

    c_left, c_right = st.columns([1.2, 1])

    with c_left:
        st.markdown("#### 📊 **Dynamic Risk Category Breakdown (Live Database Records)**")
        if not df_risk.empty and df_risk["student_count"].sum() > 0:
            fig_pie = px.pie(
                df_risk,
                names="burnout_risk",
                values="student_count",
                color="burnout_risk",
                color_discrete_map=color_map,
                hole=0.48,
                custom_data=["percentage", "avg_score"],
            )
            fig_pie.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="<b>%{label} Risk</b><br>Students: %{value} (%{customdata[0]}%)<br>Avg Score: %{customdata[1]:.1f}/100<extra></extra>",
            )
            fig_pie.update_layout(
                template=plotly_layout["template"],
                paper_bgcolor=plotly_layout["paper_bgcolor"],
                plot_bgcolor=plotly_layout["plot_bgcolor"],
                font_color=plotly_layout["font_color"],
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                margin=dict(t=10, b=30, l=10, r=10),
            )
            st.plotly_chart(fig_pie)
        else:
            st.info("ℹ️ **No student assessments recorded yet.** When students submit assessments, live risk distributions will automatically render here.")

    with c_right:
        st.markdown("#### 📋 **Institutional Cohort Metrics**")
        sleep_str = f"{kpis['avg_sleep_hours']} hrs/night" if kpis['total_assessments'] > 0 else "0.0 hrs"
        press_str = f"{kpis['avg_academic_pressure']} / 100" if kpis['total_assessments'] > 0 else "0.0 / 100"
        bal_str = f"{kpis['avg_lifestyle_balance']} / 100" if kpis['total_assessments'] > 0 else "0.0 / 100"

        theme = get_current_theme()
        box_bg = "#1e293b" if theme == "dark" else "#f1f5f9"
        box_border = "#334155" if theme == "dark" else "#cbd5e1"
        text_color = "#f8fafc" if theme == "dark" else "#0f172a"

        st.markdown(
            f"""
            <div style="background:{box_bg}; border:1px solid {box_border}; border-radius:10px; padding:1.25rem;">
                <p style="color:{text_color}; margin-bottom:0.8rem;">
                    <strong>Cohort Average Sleep:</strong> <span style="color:#38bdf8; font-weight:600;">{sleep_str}</span>
                </p>
                <p style="color:{text_color}; margin-bottom:0.8rem;">
                    <strong>Academic Pressure Index:</strong> <span style="color:#fb923c; font-weight:600;">{press_str}</span>
                </p>
                <p style="color:{text_color}; margin-bottom:0.8rem;">
                    <strong>Lifestyle Balance Index:</strong> <span style="color:#4ade80; font-weight:600;">{bal_str}</span>
                </p>
                <p style="color:{text_color}; margin-bottom:0;">
                    <strong>High & Very High Risk Count:</strong> <span style="color:#ef4444; font-weight:600;">{kpis['high_risk_count']} students</span>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### ⚡ Navigation Shortcuts")
        qa1, qa2 = st.columns(2)
        with qa1:
            st.markdown(
                """
                **📊 [Population Analytics](Population_Analytics)**  
                Inspect multi-dimensional cohort distributions and SQL queries.
                """
            )
        with qa2:
            st.markdown(
                """
                **🔬 [Model Insights](Model_Insights)**  
                Evaluate ML metrics, confusion matrices, and feature influence.
                """
            )

    st.markdown("---")

    # Dynamic Institutional SWOT Analysis Section
    st.markdown("### 🎯 **Institutional SWOT Diagnostics (Derived from Actual DB Data)**")
    st.caption("Automated synthesis of cohort strengths, vulnerabilities, and proactive intervention opportunities:")

    swot = get_admin_swot_analysis()
    if not swot["available"]:
        st.info(f"ℹ️ {swot['message']}")
    else:
        sw1, sw2, sw3, sw4 = st.columns(4)
        with sw1:
            st.markdown(
                f"""
                <div class="swot-card" style="border-top: 4px solid #22c55e;">
                    <div class="swot-title" style="color:#22c55e;">💪 Strengths</div>
                    <ul style="font-size:0.85rem; padding-left:1.1rem; margin:0;">
                        {''.join([f'<li style="margin-bottom:0.4rem;">{s}</li>' for s in swot['strengths']])}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sw2:
            st.markdown(
                f"""
                <div class="swot-card" style="border-top: 4px solid #f97316;">
                    <div class="swot-title" style="color:#f97316;">⚠️ Weaknesses</div>
                    <ul style="font-size:0.85rem; padding-left:1.1rem; margin:0;">
                        {''.join([f'<li style="margin-bottom:0.4rem;">{w}</li>' for w in swot['weaknesses']])}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sw3:
            st.markdown(
                f"""
                <div class="swot-card" style="border-top: 4px solid #38bdf8;">
                    <div class="swot-title" style="color:#38bdf8;">🌟 Opportunities</div>
                    <ul style="font-size:0.85rem; padding-left:1.1rem; margin:0;">
                        {''.join([f'<li style="margin-bottom:0.4rem;">{o}</li>' for o in swot['opportunities']])}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with sw4:
            st.markdown(
                f"""
                <div class="swot-card" style="border-top: 4px solid #ef4444;">
                    <div class="swot-title" style="color:#ef4444;">🚨 Threats</div>
                    <ul style="font-size:0.85rem; padding-left:1.1rem; margin:0;">
                        {''.join([f'<li style="margin-bottom:0.4rem;">{t}</li>' for t in swot['threats']])}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Searchable Student Assessment Records
    st.markdown("### 📋 **Student Assessment Records**")
    st.caption("Privacy-conscious student records computed strictly from active database submissions:")

    search_term = st.text_input("🔍 Search Student Records by Name or Department:", placeholder="e.g. Aditi, Computer Science")
    df_records = get_admin_student_records(search_query=search_term)

    if df_records.empty:
        st.info("No student assessment records found matching the query.")
    else:
        display_df = df_records[[
            "name", "course", "year_of_study", "assessment_date",
            "sleep_hours", "study_hours_per_day", "academic_pressure_score",
            "lifestyle_balance_score", "burnout_score", "risk_category"
        ]].copy()
        display_df.columns = [
            "Student Name", "Department", "Year", "Submitted At",
            "Sleep (h)", "Study (h/d)", "Academic Strain", "Lifestyle Buffer",
            "Burnout Score", "Risk Tier"
        ]
        st.dataframe(display_df, hide_index=True)

    render_disclaimer()

# =====================================================================
# 3. AUTHENTICATED STUDENT DASHBOARD (Role: Student)
# =====================================================================
else:
    user = get_current_user()
    profile = get_current_profile() or {}

    user_id = user["user_id"]
    full_name = profile.get("full_name", user.get("name", "Student"))
    course = profile.get("course", "Computer Science")
    year = profile.get("year_of_study", 1)

    render_header(
        title="Home",
        subtitle=f"Welcome back, {full_name} 👋 · {course} (Year {year})",
        icon="🎓",
    )

    # Fetch latest prediction and history for this student
    latest_pred = get_latest_user_prediction(user_id)
    history = get_user_assessment_history(user_id)

    if not latest_pred:
        # First-Time Student Onboarding Banner
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                        border: 1px solid #38bdf8; border-radius: 12px; padding: 2rem; text-align: center; margin-bottom: 2rem;">
                <h2 style="color:#f8fafc; margin-top:0;">👋 Welcome to MindMap AI</h2>
                <p style="color:#cbd5e1; max-width:650px; margin: 0.5rem auto 1.5rem auto; font-size:1.05rem;">
                    Complete your first assessment to receive your personalized burnout risk analysis, 
                    explainable contributing factors, and data-driven recommendations.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 📋 **Your Academic Profile**")
        prof_col1, prof_col2, prof_col3 = st.columns(3)
        with prof_col1:
            st.metric("Course / Department", course)
        with prof_col2:
            st.metric("Year of Study", f"Year {year}")
        with prof_col3:
            st.metric("Assessments Completed", "0")

        st.markdown("###")
        st.info("👉 Click **My Assessment** in the sidebar to start your first evaluation!")

    else:
        # Full Personalized Dashboard
        badge_html = get_risk_badge_html(latest_pred["burnout_risk"])
        last_date = latest_pred["created_at"][:10]

        # 4 Primary KPI Summary Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Current Burnout Risk</div>
                    <div class="kpi-value">{latest_pred['risk_score']} <span style="font-size:1rem;color:#94a3b8">/ 100</span></div>
                    <div>{badge_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Academic Strain Index</div>
                    <div class="kpi-value" style="color:#ef4444">{latest_pred['academic_pressure_score']} <span style="font-size:1rem;color:#94a3b8">/ 100</span></div>
                    <div class="kpi-subtitle">Workload & exam pressure</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Lifestyle Balance Index</div>
                    <div class="kpi-value" style="color:#38bdf8">{latest_pred['lifestyle_balance_score']} <span style="font-size:1rem;color:#94a3b8">/ 100</span></div>
                    <div class="kpi-subtitle">Sleep, exercise & recovery buffer</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Assessments Completed</div>
                    <div class="kpi-value">{len(history)} <span style="font-size:1rem;color:#94a3b8">logged</span></div>
                    <div class="kpi-subtitle">Last evaluation: {last_date}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("###")

        # Two Column Layout: Contributing Factors vs Latest Recommendations
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### 🔍 **Key Contributing Factors**")
            st.caption("Factors exerting the strongest model influence relative to baseline benchmarks:")

            factors = latest_pred.get("contributing_factors", [])
            if factors:
                for f in factors[:3]:
                    direction_class = "elevates" if f.get("direction") == "elevates_risk" else "lowers"
                    icon = "🔺" if f.get("direction") == "elevates_risk" else "🛡️"
                    st.markdown(
                        f"""
                        <div class="factor-card {direction_class}">
                            <div style="font-weight:600; font-size:0.95rem; margin-bottom:0.25rem;">
                                {icon} {f.get('label', f.get('feature', ''))}
                            </div>
                            <div style="font-size:0.875rem;">
                                {f.get('explanation', '')}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Your inputs match the student cohort baseline closely.")

        with col_right:
            st.markdown("#### 💡 **Personalized Pacing Recommendations**")
            st.caption("Data-driven behavioral observations from your latest assessment:")

            recs = latest_pred.get("recommendations", [])
            if recs:
                for rec in recs[:2]:
                    priority_color = "#ef4444" if rec.get("priority") == "High Impact" else "#38bdf8"
                    st.markdown(
                        f"""
                        <div class="factor-card" style="border-left: 4px solid {priority_color}; margin-bottom:0.75rem;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem;">
                                <span style="font-weight:600; font-size:0.95rem;">
                                    {rec.get('icon', '💡')} {rec.get('title', '')}
                                </span>
                                <span style="font-size:0.75rem; padding:0.15rem 0.5rem; border-radius:4px; background:rgba(255,255,255,0.1); color:{priority_color};">
                                    {rec.get('priority', '')}
                                </span>
                            </div>
                            <p style="font-size:0.85rem; color:#94a3b8; margin:0.2rem 0 0.35rem 0;">
                                {rec.get('observation', '')}
                            </p>
                            <p style="font-size:0.85rem; color:#38bdf8; margin:0;">
                                <strong>Action:</strong> {rec.get('suggested_action', '')}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.success("Your current lifestyle and workload parameters reflect a balanced profile!")

        st.markdown("---")

        # Longitudinal Risk History Trend (if >= 2 assessments exist)
        if len(history) >= 2:
            st.markdown("### 📈 **Your Burnout Risk History & Trend**")
            st.caption("Track how changes in your schedule and habits impacted your risk score over time:")

            dates = [h["created_at"][:16] for h in history]
            scores = [h["risk_score"] for h in history]
            pressures = [h["academic_pressure_score"] for h in history]
            balances = [h["lifestyle_balance_score"] for h in history]

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=dates, y=scores, mode="lines+markers",
                name="Burnout Risk Score",
                line=dict(color="#ef4444", width=3),
                marker=dict(size=8),
            ))
            fig_trend.add_trace(go.Scatter(
                x=dates, y=pressures, mode="lines+markers",
                name="Academic Pressure Score",
                line=dict(color="#fb923c", width=2, dash="dot"),
                marker=dict(size=6),
            ))
            fig_trend.add_trace(go.Scatter(
                x=dates, y=balances, mode="lines+markers",
                name="Lifestyle Balance Score",
                line=dict(color="#38bdf8", width=2, dash="dash"),
                marker=dict(size=6),
            ))

            fig_trend.update_layout(
                template=plotly_layout["template"],
                paper_bgcolor=plotly_layout["paper_bgcolor"],
                plot_bgcolor=plotly_layout["plot_bgcolor"],
                font_color=plotly_layout["font_color"],
                height=350,
                margin=dict(t=20, b=20, l=20, r=20),
                yaxis=dict(range=[0, 100], title="Score (0–100)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_trend)
        else:
            st.info("📈 **Risk Trend:** Submit more assessments over time to track your longitudinal risk trajectory.")

        # Quick navigation actions
        st.markdown("### ⚡ **Quick Student Actions**")
        qa1, qa2, qa3 = st.columns(3)
        with qa1:
            st.markdown(
                """
                **🎯 [Take New Assessment](My_Assessment)**  
                Update your current sleep and workload metrics to recalculate risk.
                """
            )
        with qa2:
            st.markdown(
                """
                **🔮 [Open What-If Simulator](What_If_Simulator)**  
                Simulate potential lifestyle habit adjustments against your baseline.
                """
            )
        with qa3:
            st.markdown(
                """
                **👤 [Edit Student Profile](Profile)**  
                Update your enrolled course, year of study, CGPA, or attendance.
                """
            )

    render_disclaimer()
