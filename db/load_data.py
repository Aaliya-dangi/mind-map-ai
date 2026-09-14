"""
MindMap AI - Database Loader Script
Populates the 4-table normalized relational schema from the cleaned dataset.
Idempotent and safely re-runnable per design.md §27 and RMD.md §6.
"""

import sys
import sqlite3
from pathlib import Path
import pandas as pd
import mysql.connector

# Add project root to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from db.connection import get_mysql_connection, get_sqlite_connection, SQLITE_DB_PATH


def load_into_sqlite(df: pd.DataFrame) -> dict:
    """
    Idempotently creates schema and populates SQLite database for zero-config local operations.
    """
    conn = get_sqlite_connection()
    cursor = conn.cursor()

    # Create tables
    cursor.executescript("""
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS students (
        student_id TEXT PRIMARY KEY,
        age INTEGER NOT NULL,
        gender TEXT NOT NULL,
        year_of_study INTEGER NOT NULL,
        course TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS academic_data (
        student_id TEXT PRIMARY KEY,
        attendance_percentage REAL NOT NULL,
        cgpa REAL NOT NULL,
        study_hours_per_day REAL NOT NULL,
        assignment_workload INTEGER NOT NULL,
        exam_frequency INTEGER NOT NULL,
        academic_pressure INTEGER NOT NULL,
        FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS lifestyle_data (
        student_id TEXT PRIMARY KEY,
        sleep_hours REAL NOT NULL,
        sleep_quality INTEGER NOT NULL,
        screen_time_hours REAL NOT NULL,
        physical_activity_hours REAL NOT NULL,
        social_interaction_hours REAL NOT NULL,
        breaks_per_day INTEGER NOT NULL,
        hobbies_hours_per_week REAL NOT NULL,
        days_off_per_week INTEGER NOT NULL,
        FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS burnout_predictions (
        student_id TEXT PRIMARY KEY,
        academic_pressure_score REAL NOT NULL,
        lifestyle_balance_score REAL NOT NULL,
        burnout_score REAL NOT NULL,
        burnout_risk TEXT NOT NULL,
        model_version TEXT DEFAULT 'v1.0.0',
        predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
    );
    """)

    # Split DataFrame into the 4 table schemas
    df_students = df[["student_id", "age", "gender", "year_of_study", "course"]]
    df_academic = df[[
        "student_id", "attendance_percentage", "cgpa", "study_hours_per_day",
        "assignment_workload", "exam_frequency", "academic_pressure"
    ]]
    df_lifestyle = df[[
        "student_id", "sleep_hours", "sleep_quality", "screen_time_hours",
        "physical_activity_hours", "social_interaction_hours",
        "breaks_per_day", "hobbies_hours_per_week", "days_off_per_week"
    ]]
    df_predictions = df[[
        "student_id", "academic_pressure_score", "lifestyle_balance_score",
        "burnout_score", "burnout_risk"
    ]].copy()
    df_predictions["model_version"] = "v1.0.0"

    # Replace / insert data
    df_students.to_sql("students", conn, if_exists="replace", index=False)
    df_academic.to_sql("academic_data", conn, if_exists="replace", index=False)
    df_lifestyle.to_sql("lifestyle_data", conn, if_exists="replace", index=False)
    df_predictions.to_sql("burnout_predictions", conn, if_exists="replace", index=False)

    conn.commit()

    # Verify counts
    counts = {}
    for tbl in ["students", "academic_data", "lifestyle_data", "burnout_predictions"]:
        cursor.execute(f"SELECT COUNT(*) FROM {tbl}")
        counts[tbl] = cursor.fetchone()[0]

    conn.close()
    return counts


def load_into_mysql(df: pd.DataFrame) -> dict:
    """
    Idempotently creates schema and populates MySQL database.
    """
    conn = get_mysql_connection(create_db_if_missing=True)
    if conn is None or not conn.is_connected():
        raise ConnectionError("Could not connect to MySQL server.")

    cursor = conn.cursor()

    # Read and execute schema.sql
    schema_path = Path(__file__).resolve().parent / "schema.sql"
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    # Execute statements
    for statement in schema_sql.split(";"):
        stmt = statement.strip()
        if stmt:
            cursor.execute(stmt)

    # Insert statements using REPLACE INTO for idempotency
    students_records = df[["student_id", "age", "gender", "year_of_study", "course"]].values.tolist()
    cursor.executemany(
        "REPLACE INTO students (student_id, age, gender, year_of_study, course) VALUES (%s, %s, %s, %s, %s)",
        students_records,
    )

    academic_records = df[[
        "student_id", "attendance_percentage", "cgpa", "study_hours_per_day",
        "assignment_workload", "exam_frequency", "academic_pressure"
    ]].values.tolist()
    cursor.executemany(
        """REPLACE INTO academic_data 
           (student_id, attendance_percentage, cgpa, study_hours_per_day, assignment_workload, exam_frequency, academic_pressure) 
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        academic_records,
    )

    lifestyle_records = df[[
        "student_id", "sleep_hours", "sleep_quality", "screen_time_hours",
        "physical_activity_hours", "social_interaction_hours",
        "breaks_per_day", "hobbies_hours_per_week", "days_off_per_week"
    ]].values.tolist()
    cursor.executemany(
        """REPLACE INTO lifestyle_data 
           (student_id, sleep_hours, sleep_quality, screen_time_hours, physical_activity_hours, social_interaction_hours, breaks_per_day, hobbies_hours_per_week, days_off_per_week) 
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        lifestyle_records,
    )

    pred_records = df[[
        "student_id", "academic_pressure_score", "lifestyle_balance_score",
        "burnout_score", "burnout_risk"
    ]].values.tolist()
    pred_records = [rec + ["v1.0.0"] for rec in pred_records]
    cursor.executemany(
        """REPLACE INTO burnout_predictions 
           (student_id, academic_pressure_score, lifestyle_balance_score, burnout_score, burnout_risk, model_version) 
           VALUES (%s, %s, %s, %s, %s, %s)""",
        pred_records,
    )

    conn.commit()

    # Verify counts
    counts = {}
    for tbl in ["students", "academic_data", "lifestyle_data", "burnout_predictions"]:
        cursor.execute(f"SELECT COUNT(*) FROM {tbl}")
        counts[tbl] = cursor.fetchone()[0]

    cursor.close()
    conn.close()
    return counts


def load_database(dataset_path: Path = config.PROCESSED_DATA_FILE) -> dict:
    """
    Loads dataset into database (SQLite guaranteed fallback + MySQL if available).
    """
    print(f"Loading data from {dataset_path} into relational database...")
    df = pd.read_csv(dataset_path)

    # 1. Populate SQLite database (ensures 100% offline & zero-config availability)
    sqlite_counts = load_into_sqlite(df)
    print(f"[OK] SQLite database successfully loaded at {SQLITE_DB_PATH}:")
    for tbl, count in sqlite_counts.items():
        print(f"     - Table '{tbl}': {count} rows")

    # 2. Try populating MySQL if accessible
    mysql_status = "Skipped (MySQL credentials not provided or server offline)"
    try:
        mysql_counts = load_into_mysql(df)
        print(f"[OK] MySQL database '{config.DB_NAME}' successfully loaded:")
        for tbl, count in mysql_counts.items():
            print(f"     - Table '{tbl}': {count} rows")
        mysql_status = "Loaded successfully"
    except Exception as e:
        print(f"[INFO] MySQL loader note: {e}")
        print("       (Automatic fallback to SQLite ensures all SQL queries & dashboard features work seamlessly.)")

    return {
        "sqlite_counts": sqlite_counts,
        "mysql_status": mysql_status,
    }


def main():
    load_database()


if __name__ == "__main__":
    main()
