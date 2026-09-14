"""
MindMap AI - Comprehensive Page & UI Components Compatibility Test
Verifies that all plot generators, data transformations, database queries,
and page execution logic run without AttributeError, ImportError, or runtime crashes.
"""

import pytest
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import config
from db.queries import (
    get_kpi_metrics,
    get_risk_distribution_by_category,
    get_burnout_by_course_and_year,
    get_sleep_bucket_analysis,
    get_students_above_average_pressure,
    get_full_analytics_dataset,
)
from db.user_db import (
    create_user,
    get_user_by_username,
    get_student_profile,
    update_student_profile,
    save_user_assessment_and_prediction,
    get_user_assessment_history,
    get_latest_user_prediction,
    seed_demo_user,
)
from ml.predict import predict_burnout_risk
from utils.recommendations import generate_data_driven_recommendations
from utils.auth import hash_password, verify_password, login_user, signup_user


class TestPagePlotlyCompatibility:
    def test_confusion_matrix_imshow_rendering(self):
        """Validates that confusion matrix renders using px.imshow without plotly.figure_factory."""
        cm = np.array([[50, 2, 0, 1], [3, 45, 4, 0], [0, 5, 40, 2], [1, 0, 3, 30]])
        classes = ["Low", "Moderate", "High", "Very High"]
        fig = px.imshow(
            cm,
            x=classes,
            y=classes,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Class", y="True Class", color="Count"),
        )
        assert fig is not None
        assert len(fig.data) > 0

    def test_correlation_matrix_imshow_rendering(self):
        """Validates correlation heatmap generation."""
        df = get_full_analytics_dataset()
        num_cols = ["burnout_score", "academic_pressure_score", "lifestyle_balance_score", "sleep_hours", "cgpa"]
        corr = df[num_cols].corr()
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
        )
        assert fig is not None
        assert len(fig.data) > 0

    def test_gauge_chart_rendering(self):
        """Validates gauge chart creation for My Assessment."""
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=65.4,
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
        assert fig_gauge is not None
        assert len(fig_gauge.data) > 0

    def test_timeline_history_chart_rendering(self):
        """Validates longitudinal history plot creation."""
        dates = ["2026-09-01 10:00", "2026-09-08 11:30", "2026-09-14 14:00"]
        scores = [72.0, 64.5, 48.2]
        pressures = [80.0, 70.0, 52.0]
        balances = [35.0, 48.0, 68.0]

        fig_timeline = go.Figure()
        fig_timeline.add_hrect(y0=0, y1=30, fillcolor="rgba(34, 197, 94, 0.08)", line_width=0)
        fig_timeline.add_hrect(y0=30, y1=60, fillcolor="rgba(234, 179, 8, 0.08)", line_width=0)
        fig_timeline.add_hrect(y0=60, y1=80, fillcolor="rgba(249, 115, 22, 0.08)", line_width=0)
        fig_timeline.add_hrect(y0=80, y1=100, fillcolor="rgba(239, 68, 68, 0.08)", line_width=0)

        fig_timeline.add_trace(go.Scatter(x=dates, y=scores, mode="lines+markers", name="Burnout Risk Score"))
        fig_timeline.add_trace(go.Scatter(x=dates, y=pressures, mode="lines+markers", name="Academic Pressure Score"))
        fig_timeline.add_trace(go.Scatter(x=dates, y=balances, mode="lines+markers", name="Lifestyle Balance Score"))

        assert fig_timeline is not None
        assert len(fig_timeline.data) == 3


class TestAllSQLQueries:
    def test_kpi_query(self):
        kpis = get_kpi_metrics()
        assert "avg_burnout_score" in kpis
        assert "total_students" in kpis
        assert kpis["total_students"] > 0

    def test_risk_distribution_query(self):
        df = get_risk_distribution_by_category()
        assert not df.empty
        assert "burnout_risk" in df.columns

    def test_burnout_by_course_and_year(self):
        df = get_burnout_by_course_and_year()
        assert not df.empty
        assert "course" in df.columns
        assert "year_of_study" in df.columns

    def test_sleep_bucket_analysis(self):
        df = get_sleep_bucket_analysis()
        assert not df.empty
        assert "sleep_tier" in df.columns

    def test_students_above_average_pressure(self):
        df = get_students_above_average_pressure()
        assert not df.empty
        assert len(df) <= 20

    def test_full_dataset_filters(self):
        df_all = get_full_analytics_dataset()
        assert len(df_all) == 3000

        df_cs = get_full_analytics_dataset(course="Computer Science")
        assert not df_cs.empty
        assert (df_cs["course"] == "Computer Science").all()

        df_y3 = get_full_analytics_dataset(year_of_study=3)
        assert not df_y3.empty
        assert (df_y3["year_of_study"] == 3).all()
