"""
MindMap AI - SQL Analytics Validation & Spot-Check Suite
Validates that SQL query results match equivalent Pandas computations
per RMD.md §4 and §17.
"""

import pytest
import pandas as pd
import numpy as np

import config
from db.queries import (
    get_kpi_metrics,
    get_risk_distribution_by_category,
    get_sleep_bucket_analysis,
    get_students_above_average_pressure,
)


@pytest.fixture(scope="module")
def df_clean():
    """Loads cleaned dataset for benchmark comparisons."""
    return pd.read_csv(config.PROCESSED_DATA_FILE)


class TestSQLAnalyticsValidation:
    def test_kpi_metrics_match_pandas(self, df_clean):
        """Spot-check SQL KPI aggregations against Pandas dataframe calculations."""
        sql_kpis = get_kpi_metrics()

        pd_total = len(df_clean)
        pd_avg_burnout = round(float(df_clean["burnout_score"].mean()), 1)
        pd_avg_sleep = round(float(df_clean["sleep_hours"].mean()), 1)
        pd_avg_pressure = round(float(df_clean["academic_pressure_score"].mean()), 1)
        pd_high_risk_pct = round(float(df_clean["burnout_risk"].isin(["High", "Very High"]).mean() * 100), 1)

        assert sql_kpis["total_students"] == pd_total
        assert abs(sql_kpis["avg_burnout_score"] - pd_avg_burnout) <= 0.2
        assert abs(sql_kpis["avg_sleep_hours"] - pd_avg_sleep) <= 0.2
        assert abs(sql_kpis["avg_academic_pressure"] - pd_avg_pressure) <= 0.2
        assert abs(sql_kpis["high_risk_percentage"] - pd_high_risk_pct) <= 0.2

    def test_risk_distribution_matches_pandas(self, df_clean):
        """Spot-check SQL risk category counts against Pandas value_counts."""
        sql_risk_df = get_risk_distribution_by_category()
        pd_risk_counts = df_clean["burnout_risk"].value_counts().to_dict()

        for _, row in sql_risk_df.iterrows():
            cat = row["burnout_risk"]
            count = int(row["student_count"])
            assert count == pd_risk_counts[cat]

    def test_sleep_bucketing_matches_pandas(self, df_clean):
        """Spot-check SQL CASE statement sleep tiers against Pandas conditional bucketing."""
        sql_sleep_tiers = get_sleep_bucket_analysis()

        # Severe Sleep (< 5.5h)
        pd_severe = df_clean[df_clean["sleep_hours"] < 5.5]
        sql_severe = sql_sleep_tiers[sql_sleep_tiers["sleep_tier"].str.contains("< 5.5h")].iloc[0]
        assert int(sql_severe["student_count"]) == len(pd_severe)

        # Optimal Sleep (> 8.5h)
        pd_optimal = df_clean[df_clean["sleep_hours"] >= 8.5]
        sql_optimal = sql_sleep_tiers[sql_sleep_tiers["sleep_tier"].str.contains("> 8.5h")].iloc[0]
        assert int(sql_optimal["student_count"]) == len(pd_optimal)

    def test_subquery_above_average_pressure_matches_pandas(self, df_clean):
        """Spot-check SQL subquery against Pandas filtering for pressure > mean."""
        sql_above_avg = get_students_above_average_pressure()
        mean_pressure = df_clean["academic_pressure_score"].mean()
        pd_above_avg = df_clean[df_clean["academic_pressure_score"] > mean_pressure]

        # Check top student from SQL subquery is indeed the highest pressure score in Pandas
        top_sql_score = float(sql_above_avg.iloc[0]["academic_pressure_score"])
        top_pd_score = float(pd_above_avg["academic_pressure_score"].max())
        assert abs(top_sql_score - top_pd_score) <= 0.05
