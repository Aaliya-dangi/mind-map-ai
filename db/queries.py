"""
MindMap AI - SQL Analytics Queries Module
Implements SQL analytical queries using JOINs, GROUP BY, CASE statements,
aggregations, and subqueries per design.md §28 and RMD.md §7-8.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd

# Add project root to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from db.connection import execute_query


def get_kpi_metrics(
    course: Optional[str] = None,
    year_of_study: Optional[int] = None,
    gender: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Computes top-level KPI metrics using SQL aggregation:
    - Average Burnout Risk Score
    - High / Very High Risk Student %
    - Average Sleep Hours
    - Average Academic Pressure Score
    """
    where_clauses = []
    params = []

    if course and course != "All":
        where_clauses.append("s.course = ?")
        params.append(course)
    if year_of_study and year_of_study != "All":
        where_clauses.append("s.year_of_study = ?")
        params.append(int(year_of_study))
    if gender and gender != "All":
        where_clauses.append("s.gender = ?")
        params.append(gender)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    query = f"""
    SELECT 
        COUNT(s.student_id) AS total_students,
        AVG(p.burnout_score) AS avg_burnout_score,
        AVG(l.sleep_hours) AS avg_sleep_hours,
        AVG(p.academic_pressure_score) AS avg_academic_pressure,
        AVG(p.lifestyle_balance_score) AS avg_lifestyle_balance,
        SUM(CASE WHEN p.burnout_risk IN ('High', 'Very High') THEN 1 ELSE 0 END) * 100.0 / COUNT(s.student_id) AS high_risk_percentage
    FROM students s
    JOIN academic_data a ON s.student_id = a.student_id
    JOIN lifestyle_data l ON s.student_id = l.student_id
    JOIN burnout_predictions p ON s.student_id = p.student_id
    {where_sql}
    """

    df, meta = execute_query(query, tuple(params) if params else None)
    if df.empty:
        return {
            "total_students": 0,
            "avg_burnout_score": 0.0,
            "avg_sleep_hours": 0.0,
            "avg_academic_pressure": 0.0,
            "avg_lifestyle_balance": 0.0,
            "high_risk_percentage": 0.0,
            "engine": meta.get("engine", "unknown"),
        }

    row = df.iloc[0]
    return {
        "total_students": int(row["total_students"]),
        "avg_burnout_score": round(float(row["avg_burnout_score"] or 0), 1),
        "avg_sleep_hours": round(float(row["avg_sleep_hours"] or 0), 1),
        "avg_academic_pressure": round(float(row["avg_academic_pressure"] or 0), 1),
        "avg_lifestyle_balance": round(float(row["avg_lifestyle_balance"] or 0), 1),
        "high_risk_percentage": round(float(row["high_risk_percentage"] or 0), 1),
        "engine": meta.get("engine", "unknown"),
    }


def get_risk_distribution_by_category(
    course: Optional[str] = None,
    year_of_study: Optional[int] = None,
    gender: Optional[str] = None,
) -> pd.DataFrame:
    """
    SQL Query: Risk category counts and percentages.
    Uses: SELECT, JOIN, WHERE, GROUP BY, ORDER BY, CASE.
    """
    where_clauses = []
    params = []

    if course and course != "All":
        where_clauses.append("s.course = ?")
        params.append(course)
    if year_of_study and year_of_study != "All":
        where_clauses.append("s.year_of_study = ?")
        params.append(int(year_of_study))
    if gender and gender != "All":
        where_clauses.append("s.gender = ?")
        params.append(gender)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    query = f"""
    SELECT 
        p.burnout_risk,
        COUNT(s.student_id) AS student_count,
        ROUND(COUNT(s.student_id) * 100.0 / (SELECT COUNT(*) FROM students), 2) AS percentage,
        AVG(p.burnout_score) AS avg_score
    FROM students s
    JOIN burnout_predictions p ON s.student_id = p.student_id
    {where_sql}
    GROUP BY p.burnout_risk
    ORDER BY 
        CASE p.burnout_risk
            WHEN 'Low' THEN 1
            WHEN 'Moderate' THEN 2
            WHEN 'High' THEN 3
            WHEN 'Very High' THEN 4
            ELSE 5
        END
    """
    df, _ = execute_query(query, tuple(params) if params else None)
    return df


def get_burnout_by_course_and_year() -> pd.DataFrame:
    """
    SQL Query: Aggregates risk and scores by course and year of study.
    Uses: SELECT, JOIN, GROUP BY, AVG, CASE.
    """
    query = """
    SELECT 
        s.course,
        s.year_of_study,
        COUNT(s.student_id) AS student_count,
        ROUND(AVG(p.burnout_score), 2) AS avg_burnout_score,
        ROUND(AVG(p.academic_pressure_score), 2) AS avg_academic_pressure,
        ROUND(AVG(p.lifestyle_balance_score), 2) AS avg_lifestyle_balance,
        ROUND(AVG(l.sleep_hours), 2) AS avg_sleep_hours,
        ROUND(SUM(CASE WHEN p.burnout_risk IN ('High', 'Very High') THEN 1 ELSE 0 END) * 100.0 / COUNT(s.student_id), 2) AS high_risk_pct
    FROM students s
    JOIN academic_data a ON s.student_id = a.student_id
    JOIN lifestyle_data l ON s.student_id = l.student_id
    JOIN burnout_predictions p ON s.student_id = p.student_id
    GROUP BY s.course, s.year_of_study
    ORDER BY s.course, s.year_of_study
    """
    df, _ = execute_query(query)
    return df


def get_sleep_bucket_analysis() -> pd.DataFrame:
    """
    SQL Query: Demonstrates CASE-based bucketing of sleep duration vs burnout risk.
    Uses: SELECT, JOIN, CASE, GROUP BY, AVG, COUNT.
    """
    query = """
    SELECT 
        CASE 
            WHEN l.sleep_hours < 5.5 THEN 'Severe Sleep Loss (< 5.5h)'
            WHEN l.sleep_hours >= 5.5 AND l.sleep_hours < 7.0 THEN 'Moderate Sleep (5.5 - 7h)'
            WHEN l.sleep_hours >= 7.0 AND l.sleep_hours < 8.5 THEN 'Recommended Sleep (7 - 8.5h)'
            ELSE 'Optimal Sleep (> 8.5h)'
        END AS sleep_tier,
        COUNT(s.student_id) AS student_count,
        ROUND(AVG(l.sleep_hours), 2) AS avg_sleep,
        ROUND(AVG(p.burnout_score), 2) AS avg_burnout,
        ROUND(AVG(p.academic_pressure_score), 2) AS avg_pressure,
        ROUND(AVG(p.lifestyle_balance_score), 2) AS avg_balance,
        ROUND(SUM(CASE WHEN p.burnout_risk IN ('High', 'Very High') THEN 1 ELSE 0 END) * 100.0 / COUNT(s.student_id), 2) AS high_risk_pct
    FROM students s
    JOIN lifestyle_data l ON s.student_id = l.student_id
    JOIN burnout_predictions p ON s.student_id = p.student_id
    GROUP BY sleep_tier
    ORDER BY avg_sleep ASC
    """
    df, _ = execute_query(query)
    return df


def get_students_above_average_pressure() -> pd.DataFrame:
    """
    SQL Query: Demonstrates a SQL SUBQUERY filtering students above cohort mean pressure.
    Uses: SELECT, JOIN, WHERE with Subquery, ORDER BY.
    """
    query = """
    SELECT 
        s.student_id,
        s.course,
        s.year_of_study,
        a.academic_pressure,
        a.study_hours_per_day,
        a.assignment_workload,
        p.academic_pressure_score,
        p.burnout_score,
        p.burnout_risk
    FROM students s
    JOIN academic_data a ON s.student_id = a.student_id
    JOIN burnout_predictions p ON s.student_id = p.student_id
    WHERE p.academic_pressure_score > (
        SELECT AVG(academic_pressure_score) FROM burnout_predictions
    )
    ORDER BY p.academic_pressure_score DESC
    LIMIT 20
    """
    df, _ = execute_query(query)
    return df


def get_full_analytics_dataset(
    course: Optional[str] = None,
    year_of_study: Optional[int] = None,
    gender: Optional[str] = None,
) -> pd.DataFrame:
    """
    SQL Query: Reassembles joined tabular data across all 4 relational tables for analytics views.
    Uses: SELECT, 3x JOINs, WHERE, ORDER BY.
    """
    where_clauses = []
    params = []

    if course and course != "All":
        where_clauses.append("s.course = ?")
        params.append(course)
    if year_of_study and year_of_study != "All":
        where_clauses.append("s.year_of_study = ?")
        params.append(int(year_of_study))
    if gender and gender != "All":
        where_clauses.append("s.gender = ?")
        params.append(gender)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    query = f"""
    SELECT 
        s.student_id,
        s.age,
        s.gender,
        s.year_of_study,
        s.course,
        a.attendance_percentage,
        a.cgpa,
        a.study_hours_per_day,
        a.assignment_workload,
        a.exam_frequency,
        a.academic_pressure,
        l.sleep_hours,
        l.sleep_quality,
        l.screen_time_hours,
        l.physical_activity_hours,
        l.social_interaction_hours,
        l.breaks_per_day,
        l.hobbies_hours_per_week,
        l.days_off_per_week,
        p.academic_pressure_score,
        p.lifestyle_balance_score,
        p.burnout_score,
        p.burnout_risk
    FROM students s
    JOIN academic_data a ON s.student_id = a.student_id
    JOIN lifestyle_data l ON s.student_id = l.student_id
    JOIN burnout_predictions p ON s.student_id = p.student_id
    {where_sql}
    ORDER BY s.student_id
    """
    df, _ = execute_query(query, tuple(params) if params else None)
    return df
