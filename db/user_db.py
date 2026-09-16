"""
MindMap AI - Multi-User Data Persistence & Dynamic Analytics Module
Handles SQLite and MySQL storage for user authentication (Student & Admin roles),
student profiles, real user burnout assessments, prediction history, and dynamic admin analytics.
"""

import sys
import json
import uuid
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from db.connection import get_connection, get_mysql_connection, get_sqlite_connection


def init_user_tables():
    """
    Ensures that the multi-user authentication, student profiles, assessments,
    and prediction history tables exist in the active database engine.
    """
    conn, engine = get_connection()
    cursor = conn.cursor()

    if engine == "sqlite":
        cursor.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            username TEXT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'student',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS student_profiles (
            profile_id TEXT PRIMARY KEY,
            user_id TEXT UNIQUE NOT NULL,
            full_name TEXT,
            age INTEGER NOT NULL DEFAULT 21,
            gender TEXT NOT NULL DEFAULT 'Other',
            year_of_study INTEGER NOT NULL DEFAULT 2,
            course TEXT NOT NULL DEFAULT 'Computer Science',
            attendance_percentage REAL NOT NULL DEFAULT 85.0,
            cgpa REAL NOT NULL DEFAULT 8.0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS assessments (
            assessment_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            attendance_percentage REAL NOT NULL DEFAULT 85.0,
            cgpa REAL NOT NULL DEFAULT 8.0,
            study_hours_per_day REAL NOT NULL,
            assignment_workload INTEGER NOT NULL,
            exam_frequency INTEGER NOT NULL,
            academic_pressure INTEGER NOT NULL,
            sleep_hours REAL NOT NULL,
            sleep_quality INTEGER NOT NULL,
            screen_time_hours REAL NOT NULL,
            physical_activity_hours REAL NOT NULL,
            social_interaction_hours REAL NOT NULL,
            breaks_per_day INTEGER NOT NULL,
            hobbies_hours_per_week REAL NOT NULL,
            days_off_per_week INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id TEXT PRIMARY KEY,
            assessment_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            burnout_score REAL NOT NULL,
            risk_category TEXT NOT NULL,
            academic_pressure_score REAL NOT NULL,
            lifestyle_balance_score REAL NOT NULL,
            probabilities_json TEXT NOT NULL,
            contributing_factors_json TEXT NOT NULL,
            recommendations_json TEXT NOT NULL,
            model_version TEXT DEFAULT 'v1.0.0',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (assessment_id) REFERENCES assessments(assessment_id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );
        """)
        # Safe migration for existing SQLite files if columns were missing
        for col_stmt in [
            "ALTER TABLE users ADD COLUMN name TEXT;",
            "ALTER TABLE users ADD COLUMN email TEXT;",
            "ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'student';",
            "ALTER TABLE users ADD COLUMN username TEXT;",
            "ALTER TABLE student_profiles ADD COLUMN full_name TEXT;",
            "ALTER TABLE student_profiles ADD COLUMN profile_id TEXT;",
            "ALTER TABLE student_profiles ADD COLUMN attendance_percentage REAL DEFAULT 85.0;",
            "ALTER TABLE student_profiles ADD COLUMN cgpa REAL DEFAULT 8.0;",
        ]:
            try:
                cursor.execute(col_stmt)
            except Exception:
                pass

        try:
            cursor.execute("UPDATE users SET name = username WHERE name IS NULL OR name = '';")
            cursor.execute("UPDATE users SET email = username || '@mindmap.ai' WHERE email IS NULL OR email = '';")
            cursor.execute("UPDATE users SET role = 'student' WHERE role IS NULL OR role = '';")
        except Exception:
            pass
        conn.commit()
    else:
        # MySQL engine
        tables_sql = [
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id VARCHAR(36) PRIMARY KEY,
                name VARCHAR(128) NOT NULL,
                username VARCHAR(64),
                email VARCHAR(128) UNIQUE NOT NULL,
                password_hash VARCHAR(128) NOT NULL,
                salt VARCHAR(64) NOT NULL,
                role VARCHAR(16) NOT NULL DEFAULT 'student',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS student_profiles (
                profile_id VARCHAR(36) PRIMARY KEY,
                user_id VARCHAR(36) UNIQUE NOT NULL,
                full_name VARCHAR(128),
                age INT NOT NULL DEFAULT 21,
                gender VARCHAR(16) NOT NULL DEFAULT 'Other',
                year_of_study INT NOT NULL DEFAULT 2,
                course VARCHAR(64) NOT NULL DEFAULT 'Computer Science',
                attendance_percentage DECIMAL(5, 2) NOT NULL DEFAULT 85.0,
                cgpa DECIMAL(4, 2) NOT NULL DEFAULT 8.0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS assessments (
                assessment_id VARCHAR(36) PRIMARY KEY,
                user_id VARCHAR(36) NOT NULL,
                attendance_percentage DECIMAL(5, 2) NOT NULL DEFAULT 85.0,
                cgpa DECIMAL(4, 2) NOT NULL DEFAULT 8.0,
                study_hours_per_day DECIMAL(4, 2) NOT NULL,
                assignment_workload INT NOT NULL,
                exam_frequency INT NOT NULL,
                academic_pressure INT NOT NULL,
                sleep_hours DECIMAL(4, 2) NOT NULL,
                sleep_quality INT NOT NULL,
                screen_time_hours DECIMAL(4, 2) NOT NULL,
                physical_activity_hours DECIMAL(4, 2) NOT NULL,
                social_interaction_hours DECIMAL(4, 2) NOT NULL,
                breaks_per_day INT NOT NULL,
                hobbies_hours_per_week DECIMAL(4, 2) NOT NULL,
                days_off_per_week INT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id VARCHAR(36) PRIMARY KEY,
                assessment_id VARCHAR(36) NOT NULL,
                user_id VARCHAR(36) NOT NULL,
                burnout_score DECIMAL(5, 2) NOT NULL,
                risk_category VARCHAR(16) NOT NULL,
                academic_pressure_score DECIMAL(5, 2) NOT NULL,
                lifestyle_balance_score DECIMAL(5, 2) NOT NULL,
                probabilities_json TEXT NOT NULL,
                contributing_factors_json TEXT NOT NULL,
                recommendations_json TEXT NOT NULL,
                model_version VARCHAR(32) DEFAULT 'v1.0.0',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (assessment_id) REFERENCES assessments(assessment_id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );
            """
        ]
        for tbl_sql in tables_sql:
            cursor.execute(tbl_sql)
        conn.commit()

    cursor.close()
    conn.close()


def create_user(
    name: Optional[str] = None,
    email: Optional[str] = None,
    password_hash: str = "",
    salt: str = "",
    role: str = "student",
    age: int = 21,
    gender: str = "Other",
    year_of_study: int = 2,
    course: str = "Computer Science",
    username: Optional[str] = None,
    full_name: Optional[str] = None,
    attendance_percentage: float = 85.0,
    cgpa: float = 8.0,
    **kwargs,
) -> Optional[Dict[str, Any]]:
    """
    Creates a new user and corresponding student profile if role is 'student'.
    Returns user dictionary if successful, or None if email/username already exists.
    """
    init_user_tables()
    conn, engine = get_connection()
    cursor = conn.cursor()

    user_id = str(uuid.uuid4())
    profile_id = str(uuid.uuid4())

    display_name = (name or full_name or username or "Student").strip()
    user_uname = (username or (email.split("@")[0] if email else display_name.lower().replace(" ", "_"))).strip().lower()
    email_clean = (email or f"{user_uname}@mindmap.ai").strip().lower()

    # Check for existing email or username
    try:
        if engine == "sqlite":
            cursor.execute(
                "SELECT user_id FROM users WHERE LOWER(email) = ? OR LOWER(username) = ?",
                (email_clean, user_uname),
            )
        else:
            cursor.execute(
                "SELECT user_id FROM users WHERE LOWER(email) = %s OR LOWER(username) = %s",
                (email_clean, user_uname),
            )
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return None
    except Exception:
        pass

    try:
        if engine == "sqlite":
            cursor.execute(
                "INSERT INTO users (user_id, name, username, email, password_hash, salt, role) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, display_name, user_uname, email_clean, password_hash, salt, role),
            )
            if role == "student":
                cursor.execute(
                    """
                    INSERT INTO student_profiles (profile_id, user_id, full_name, age, gender, year_of_study, course, attendance_percentage, cgpa)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (profile_id, user_id, display_name, int(age), gender, int(year_of_study), course, float(attendance_percentage), float(cgpa)),
                )
        else:
            cursor.execute(
                "INSERT INTO users (user_id, name, username, email, password_hash, salt, role) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (user_id, display_name, user_uname, email_clean, password_hash, salt, role),
            )
            if role == "student":
                cursor.execute(
                    """
                    INSERT INTO student_profiles (profile_id, user_id, full_name, age, gender, year_of_study, course, attendance_percentage, cgpa)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (profile_id, user_id, display_name, int(age), gender, int(year_of_study), course, float(attendance_percentage), float(cgpa)),
                )
        conn.commit()
        return {
            "user_id": user_id,
            "name": display_name,
            "full_name": display_name,
            "username": user_uname,
            "email": email_clean,
            "role": role,
            "age": int(age) if role == "student" else None,
            "gender": gender if role == "student" else None,
            "year_of_study": int(year_of_study) if role == "student" else None,
            "course": course if role == "student" else None,
            "attendance_percentage": float(attendance_percentage) if role == "student" else None,
            "cgpa": float(cgpa) if role == "student" else None,
        }
    except Exception:
        conn.rollback()
        return None
    finally:
        cursor.close()
        conn.close()


def get_user_by_email(email_or_identifier: str) -> Optional[Dict[str, Any]]:
    """Fetches user credentials, role, and profile by email or username."""
    init_user_tables()
    conn, engine = get_connection()
    cursor = conn.cursor()

    clean_id = email_or_identifier.strip().lower()
    query = """
    SELECT u.user_id, u.name, u.email, u.password_hash, u.salt, u.role,
           p.profile_id, p.age, p.gender, p.year_of_study, p.course,
           p.attendance_percentage, p.cgpa, u.username, p.full_name
    FROM users u
    LEFT JOIN student_profiles p ON u.user_id = p.user_id
    WHERE LOWER(u.email) = ? OR LOWER(u.name) = ? OR LOWER(COALESCE(u.username, '')) = ?
    """ if engine == "sqlite" else """
    SELECT u.user_id, u.name, u.email, u.password_hash, u.salt, u.role,
           p.profile_id, p.age, p.gender, p.year_of_study, p.course,
           p.attendance_percentage, p.cgpa, u.username, p.full_name
    FROM users u
    LEFT JOIN student_profiles p ON u.user_id = p.user_id
    WHERE LOWER(u.email) = %s OR LOWER(u.name) = %s OR LOWER(COALESCE(u.username, '')) = %s
    """

    cursor.execute(query, (clean_id, clean_id, clean_id))
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        return None

    name = row[1]
    username = row[13] or row[2].split("@")[0]
    full_name = row[14] or name

    return {
        "user_id": row[0],
        "name": name,
        "full_name": full_name,
        "username": username,
        "email": row[2],
        "password_hash": row[3],
        "salt": row[4],
        "role": row[5],
        "profile_id": row[6],
        "age": int(row[7]) if row[7] is not None else 21,
        "gender": row[8] or "Other",
        "year_of_study": int(row[9]) if row[9] is not None else 2,
        "course": row[10] or "Computer Science",
        "attendance_percentage": float(row[11]) if row[11] is not None else 85.0,
        "cgpa": float(row[12]) if row[12] is not None else 8.0,
    }


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Alias / lookup for user by username or email."""
    return get_user_by_email(username)


def get_student_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """Fetches student profile by user_id."""
    init_user_tables()
    conn, engine = get_connection()
    cursor = conn.cursor()

    query = """
    SELECT u.user_id, u.name, u.email, u.role,
           p.profile_id, p.age, p.gender, p.year_of_study, p.course, p.updated_at,
           p.attendance_percentage, p.cgpa, u.username, p.full_name
    FROM users u
    LEFT JOIN student_profiles p ON u.user_id = p.user_id
    WHERE u.user_id = ?
    """ if engine == "sqlite" else """
    SELECT u.user_id, u.name, u.email, u.role,
           p.profile_id, p.age, p.gender, p.year_of_study, p.course, p.updated_at,
           p.attendance_percentage, p.cgpa, u.username, p.full_name
    FROM users u
    LEFT JOIN student_profiles p ON u.user_id = p.user_id
    WHERE u.user_id = %s
    """

    cursor.execute(query, (user_id,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        return None

    name = row[1]
    username = row[12] or row[2].split("@")[0]
    full_name = row[13] or name

    return {
        "user_id": row[0],
        "name": name,
        "full_name": full_name,
        "username": username,
        "email": row[2],
        "role": row[3],
        "profile_id": row[4],
        "age": int(row[5]) if row[5] is not None else 21,
        "gender": row[6] or "Other",
        "year_of_study": int(row[7]) if row[7] is not None else 2,
        "course": row[8] or "Computer Science",
        "updated_at": str(row[9]),
        "attendance_percentage": float(row[10]) if row[10] is not None else 85.0,
        "cgpa": float(row[11]) if row[11] is not None else 8.0,
    }


def update_student_profile(
    user_id: str,
    name: Optional[str] = None,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    year_of_study: Optional[int] = None,
    course: Optional[str] = None,
    full_name: Optional[str] = None,
    attendance_percentage: Optional[float] = None,
    cgpa: Optional[float] = None,
    **kwargs,
) -> bool:
    """Updates student name and profile fields."""
    init_user_tables()
    current = get_student_profile(user_id)
    if not current:
        return False

    display_name = (full_name or name or current.get("full_name") or current.get("name") or "Student").strip()
    new_age = int(age if age is not None else current.get("age", 21))
    new_gender = str(gender if gender is not None else current.get("gender", "Other"))
    new_year = int(year_of_study if year_of_study is not None else current.get("year_of_study", 2))
    new_course = str(course if course is not None else current.get("course", "Computer Science"))
    new_att = float(attendance_percentage if attendance_percentage is not None else current.get("attendance_percentage", 85.0))
    new_cgpa = float(cgpa if cgpa is not None else current.get("cgpa", 8.0))

    conn, engine = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    try:
        if engine == "sqlite":
            cursor.execute("UPDATE users SET name = ? WHERE user_id = ?", (display_name, user_id))
            cursor.execute(
                """
                UPDATE student_profiles
                SET full_name = ?, age = ?, gender = ?, year_of_study = ?, course = ?,
                    attendance_percentage = ?, cgpa = ?, updated_at = ?
                WHERE user_id = ?
                """,
                (display_name, new_age, new_gender, new_year, new_course, new_att, new_cgpa, now_str, user_id),
            )
        else:
            cursor.execute("UPDATE users SET name = %s WHERE user_id = %s", (display_name, user_id))
            cursor.execute(
                """
                UPDATE student_profiles
                SET full_name = %s, age = %s, gender = %s, year_of_study = %s, course = %s,
                    attendance_percentage = %s, cgpa = %s, updated_at = %s
                WHERE user_id = %s
                """,
                (display_name, new_age, new_gender, new_year, new_course, new_att, new_cgpa, now_str, user_id),
            )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()


def save_user_assessment_and_prediction(
    user_id: str,
    assessment_payload: Dict[str, Any],
    prediction_result: Dict[str, Any],
    recommendations_list: List[Dict[str, Any]],
) -> str:
    """
    Persists an assessment and its ML prediction outcome to the database.
    Returns the new assessment_id.
    """
    init_user_tables()
    conn, engine = get_connection()
    cursor = conn.cursor()

    assessment_id = str(uuid.uuid4())
    prediction_id = str(uuid.uuid4())
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    probs_json = json.dumps(prediction_result.get("probabilities", {}))
    factors_json = json.dumps(prediction_result.get("contributing_factors", []))
    recs_json = json.dumps(recommendations_list)

    try:
        if engine == "sqlite":
            cursor.execute(
                """
                INSERT INTO assessments (
                    assessment_id, user_id, attendance_percentage, cgpa, study_hours_per_day,
                    assignment_workload, exam_frequency, academic_pressure, sleep_hours, sleep_quality,
                    screen_time_hours, physical_activity_hours, social_interaction_hours, breaks_per_day,
                    hobbies_hours_per_week, days_off_per_week, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    assessment_id, user_id,
                    float(assessment_payload.get("attendance_percentage", 85.0)),
                    float(assessment_payload.get("cgpa", 8.0)),
                    float(assessment_payload["study_hours_per_day"]),
                    int(assessment_payload["assignment_workload"]),
                    int(assessment_payload["exam_frequency"]),
                    int(assessment_payload["academic_pressure"]),
                    float(assessment_payload["sleep_hours"]),
                    int(assessment_payload["sleep_quality"]),
                    float(assessment_payload["screen_time_hours"]),
                    float(assessment_payload["physical_activity_hours"]),
                    float(assessment_payload["social_interaction_hours"]),
                    int(assessment_payload["breaks_per_day"]),
                    float(assessment_payload["hobbies_hours_per_week"]),
                    int(assessment_payload["days_off_per_week"]),
                    now_str,
                ),
            )
            cursor.execute(
                """
                INSERT INTO predictions (
                    prediction_id, assessment_id, user_id, burnout_score, risk_category,
                    academic_pressure_score, lifestyle_balance_score, probabilities_json,
                    contributing_factors_json, recommendations_json, model_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction_id, assessment_id, user_id,
                    float(prediction_result["risk_score"]),
                    str(prediction_result["predicted_category"]),
                    float(prediction_result["academic_pressure_score"]),
                    float(prediction_result["lifestyle_balance_score"]),
                    probs_json,
                    factors_json,
                    recs_json,
                    "v1.0.0",
                    now_str,
                ),
            )
        else:
            cursor.execute(
                """
                INSERT INTO assessments (
                    assessment_id, user_id, attendance_percentage, cgpa, study_hours_per_day,
                    assignment_workload, exam_frequency, academic_pressure, sleep_hours, sleep_quality,
                    screen_time_hours, physical_activity_hours, social_interaction_hours, breaks_per_day,
                    hobbies_hours_per_week, days_off_per_week, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    assessment_id, user_id,
                    float(assessment_payload.get("attendance_percentage", 85.0)),
                    float(assessment_payload.get("cgpa", 8.0)),
                    float(assessment_payload["study_hours_per_day"]),
                    int(assessment_payload["assignment_workload"]),
                    int(assessment_payload["exam_frequency"]),
                    int(assessment_payload["academic_pressure"]),
                    float(assessment_payload["sleep_hours"]),
                    int(assessment_payload["sleep_quality"]),
                    float(assessment_payload["screen_time_hours"]),
                    float(assessment_payload["physical_activity_hours"]),
                    float(assessment_payload["social_interaction_hours"]),
                    int(assessment_payload["breaks_per_day"]),
                    float(assessment_payload["hobbies_hours_per_week"]),
                    int(assessment_payload["days_off_per_week"]),
                    now_str,
                ),
            )
            cursor.execute(
                """
                INSERT INTO predictions (
                    prediction_id, assessment_id, user_id, burnout_score, risk_category,
                    academic_pressure_score, lifestyle_balance_score, probabilities_json,
                    contributing_factors_json, recommendations_json, model_version, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    prediction_id, assessment_id, user_id,
                    float(prediction_result["risk_score"]),
                    str(prediction_result["predicted_category"]),
                    float(prediction_result["academic_pressure_score"]),
                    float(prediction_result["lifestyle_balance_score"]),
                    probs_json,
                    factors_json,
                    recs_json,
                    "v1.0.0",
                    now_str,
                ),
            )
        conn.commit()
        return assessment_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()


def get_user_assessment_history(user_id: str) -> List[Dict[str, Any]]:
    """
    Fetches the list of all historical assessments and predictions for a user,
    ordered chronologically ascending (for plotting).
    """
    init_user_tables()
    conn, engine = get_connection()
    cursor = conn.cursor()

    query = """
    SELECT 
        a.assessment_id, a.created_at,
        p.prediction_id, p.burnout_score, p.risk_category,
        p.academic_pressure_score, p.lifestyle_balance_score,
        p.probabilities_json, p.contributing_factors_json, p.recommendations_json,
        a.study_hours_per_day, a.assignment_workload, a.exam_frequency, a.academic_pressure,
        a.sleep_hours, a.sleep_quality, a.screen_time_hours, a.physical_activity_hours,
        a.social_interaction_hours, a.breaks_per_day, a.hobbies_hours_per_week, a.days_off_per_week,
        a.attendance_percentage, a.cgpa
    FROM assessments a
    JOIN predictions p ON a.assessment_id = p.assessment_id
    WHERE a.user_id = ?
    ORDER BY a.created_at ASC
    """ if engine == "sqlite" else """
    SELECT 
        a.assessment_id, a.created_at,
        p.prediction_id, p.burnout_score, p.risk_category,
        p.academic_pressure_score, p.lifestyle_balance_score,
        p.probabilities_json, p.contributing_factors_json, p.recommendations_json,
        a.study_hours_per_day, a.assignment_workload, a.exam_frequency, a.academic_pressure,
        a.sleep_hours, a.sleep_quality, a.screen_time_hours, a.physical_activity_hours,
        a.social_interaction_hours, a.breaks_per_day, a.hobbies_hours_per_week, a.days_off_per_week,
        a.attendance_percentage, a.cgpa
    FROM assessments a
    JOIN predictions p ON a.assessment_id = p.assessment_id
    WHERE a.user_id = %s
    ORDER BY a.created_at ASC
    """

    cursor.execute(query, (user_id,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    history = []
    for r in rows:
        try:
            probs = json.loads(r[7]) if r[7] else {}
        except Exception:
            probs = {}
        try:
            factors = json.loads(r[8]) if r[8] else []
        except Exception:
            factors = []
        try:
            recs = json.loads(r[9]) if r[9] else []
        except Exception:
            recs = []

        history.append({
            "assessment_id": r[0],
            "created_at": str(r[1]),
            "prediction_id": r[2],
            "risk_score": float(r[3]),
            "burnout_risk": r[4],
            "risk_category": r[4],
            "academic_pressure_score": float(r[5]),
            "lifestyle_balance_score": float(r[6]),
            "probabilities": probs,
            "contributing_factors": factors,
            "recommendations": recs,
            "study_hours_per_day": float(r[10]),
            "assignment_workload": int(r[11]),
            "exam_frequency": int(r[12]),
            "academic_pressure": int(r[13]),
            "sleep_hours": float(r[14]),
            "sleep_quality": int(r[15]),
            "screen_time_hours": float(r[16]),
            "physical_activity_hours": float(r[17]),
            "social_interaction_hours": float(r[18]),
            "breaks_per_day": int(r[19]),
            "hobbies_hours_per_week": float(r[20]),
            "days_off_per_week": int(r[21]),
            "attendance_percentage": float(r[22]) if r[22] is not None else 85.0,
            "cgpa": float(r[23]) if r[23] is not None else 8.0,
        })

    return history


def get_latest_user_prediction(user_id: str) -> Optional[Dict[str, Any]]:
    """Returns the user's most recent assessment and prediction outcome."""
    history = get_user_assessment_history(user_id)
    if not history:
        return None
    return history[-1]


# =====================================================================
# DYNAMIC ADMIN ANALYTICS QUERIES (Computed Strictly from Actual DB Data)
# =====================================================================

def get_admin_dashboard_kpis(
    course: Optional[str] = None,
    year_of_study: Optional[int] = None,
    gender: Optional[str] = None,
    risk_category: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Dynamically aggregates KPI statistics across all real submitted student data.
    """
    init_user_tables()
    conn, engine = get_connection()
    cursor = conn.cursor()

    where_clauses = ["u.role = 'student'"]
    params = []

    if course and course != "All":
        where_clauses.append("sp.course = ?" if engine == "sqlite" else "sp.course = %s")
        params.append(course)
    if year_of_study and year_of_study != "All":
        where_clauses.append("sp.year_of_study = ?" if engine == "sqlite" else "sp.year_of_study = %s")
        params.append(int(year_of_study))
    if gender and gender != "All":
        where_clauses.append("sp.gender = ?" if engine == "sqlite" else "sp.gender = %s")
        params.append(gender)
    if risk_category and risk_category != "All":
        where_clauses.append("p.risk_category = ?" if engine == "sqlite" else "p.risk_category = %s")
        params.append(risk_category)

    where_sql = f"WHERE {' AND '.join(where_clauses)}"

    query = f"""
    SELECT 
        COUNT(DISTINCT u.user_id) AS total_students,
        COUNT(a.assessment_id) AS total_assessments,
        AVG(p.burnout_score) AS avg_burnout_score,
        AVG(a.sleep_hours) AS avg_sleep_hours,
        AVG(p.academic_pressure_score) AS avg_academic_pressure,
        AVG(p.lifestyle_balance_score) AS avg_lifestyle_balance,
        SUM(CASE WHEN p.risk_category IN ('High', 'Very High') THEN 1 ELSE 0 END) AS high_risk_count,
        SUM(CASE WHEN p.risk_category = 'Low' THEN 1 ELSE 0 END) AS low_count,
        SUM(CASE WHEN p.risk_category = 'Moderate' THEN 1 ELSE 0 END) AS moderate_count,
        SUM(CASE WHEN p.risk_category = 'High' THEN 1 ELSE 0 END) AS high_count,
        SUM(CASE WHEN p.risk_category = 'Very High' THEN 1 ELSE 0 END) AS very_high_count
    FROM users u
    LEFT JOIN student_profiles sp ON u.user_id = sp.user_id
    LEFT JOIN assessments a ON u.user_id = a.user_id
    LEFT JOIN predictions p ON a.assessment_id = p.assessment_id
    {where_sql}
    """

    cursor.execute(query, tuple(params) if params else ())
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    total_students = int(row[0]) if row and row[0] is not None else 0
    total_assessments = int(row[1]) if row and row[1] is not None else 0
    avg_burnout = round(float(row[2]), 1) if row and row[2] is not None else 0.0
    avg_sleep = round(float(row[3]), 1) if row and row[3] is not None else 0.0
    avg_pressure = round(float(row[4]), 1) if row and row[4] is not None else 0.0
    avg_balance = round(float(row[5]), 1) if row and row[5] is not None else 0.0

    high_risk_count = int(row[6]) if row and row[6] is not None else 0
    high_risk_pct = round(high_risk_count * 100.0 / total_assessments, 1) if total_assessments > 0 else 0.0

    return {
        "total_students": total_students,
        "total_assessments": total_assessments,
        "avg_burnout_score": avg_burnout,
        "avg_sleep_hours": avg_sleep,
        "avg_academic_pressure": avg_pressure,
        "avg_lifestyle_balance": avg_balance,
        "high_risk_count": high_risk_count,
        "high_risk_percentage": high_risk_pct,
        "low_count": int(row[7]) if row and row[7] is not None else 0,
        "moderate_count": int(row[8]) if row and row[8] is not None else 0,
        "high_count": int(row[9]) if row and row[9] is not None else 0,
        "very_high_count": int(row[10]) if row and row[10] is not None else 0,
    }


def get_admin_risk_distribution(
    course: Optional[str] = None,
    year_of_study: Optional[int] = None,
    gender: Optional[str] = None,
) -> pd.DataFrame:
    """Returns risk category breakdown from submitted user predictions."""
    init_user_tables()
    conn, engine = get_connection()

    where_clauses = ["u.role = 'student'", "p.risk_category IS NOT NULL"]
    params = []

    if course and course != "All":
        where_clauses.append("sp.course = ?" if engine == "sqlite" else "sp.course = %s")
        params.append(course)
    if year_of_study and year_of_study != "All":
        where_clauses.append("sp.year_of_study = ?" if engine == "sqlite" else "sp.year_of_study = %s")
        params.append(int(year_of_study))
    if gender and gender != "All":
        where_clauses.append("sp.gender = ?" if engine == "sqlite" else "sp.gender = %s")
        params.append(gender)

    where_sql = f"WHERE {' AND '.join(where_clauses)}"

    query = f"""
    SELECT 
        p.risk_category AS burnout_risk,
        COUNT(p.prediction_id) AS student_count,
        AVG(p.burnout_score) AS avg_score
    FROM users u
    JOIN student_profiles sp ON u.user_id = sp.user_id
    JOIN assessments a ON u.user_id = a.user_id
    JOIN predictions p ON a.assessment_id = p.assessment_id
    {where_sql}
    GROUP BY p.risk_category
    ORDER BY 
        CASE p.risk_category
            WHEN 'Low' THEN 1
            WHEN 'Moderate' THEN 2
            WHEN 'High' THEN 3
            WHEN 'Very High' THEN 4
            ELSE 5
        END
    """

    df = pd.read_sql_query(query, conn, params=tuple(params) if params else None)
    conn.close()

    if not df.empty:
        total = df["student_count"].sum()
        df["percentage"] = (df["student_count"] * 100.0 / total).round(1) if total > 0 else 0.0
    return df


def get_admin_cohort_breakdowns() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns dynamic aggregations grouped by academic year and by course."""
    init_user_tables()
    conn, engine = get_connection()

    query_year = """
    SELECT 
        sp.year_of_study,
        COUNT(a.assessment_id) AS assessment_count,
        AVG(p.burnout_score) AS avg_burnout,
        AVG(a.sleep_hours) AS avg_sleep,
        AVG(p.academic_pressure_score) AS avg_pressure,
        AVG(p.lifestyle_balance_score) AS avg_balance,
        SUM(CASE WHEN p.risk_category IN ('High', 'Very High') THEN 1 ELSE 0 END) * 100.0 / COUNT(a.assessment_id) AS high_risk_pct
    FROM student_profiles sp
    JOIN assessments a ON sp.user_id = a.user_id
    JOIN predictions p ON a.assessment_id = p.assessment_id
    GROUP BY sp.year_of_study
    ORDER BY sp.year_of_study
    """
    df_year = pd.read_sql_query(query_year, conn)

    query_course = """
    SELECT 
        sp.course,
        COUNT(a.assessment_id) AS assessment_count,
        AVG(p.burnout_score) AS avg_burnout,
        AVG(a.sleep_hours) AS avg_sleep,
        AVG(p.academic_pressure_score) AS avg_pressure,
        AVG(p.lifestyle_balance_score) AS avg_balance,
        SUM(CASE WHEN p.risk_category IN ('High', 'Very High') THEN 1 ELSE 0 END) * 100.0 / COUNT(a.assessment_id) AS high_risk_pct
    FROM student_profiles sp
    JOIN assessments a ON sp.user_id = a.user_id
    JOIN predictions p ON a.assessment_id = p.assessment_id
    GROUP BY sp.course
    ORDER BY avg_burnout DESC
    """
    df_course = pd.read_sql_query(query_course, conn)
    conn.close()

    return df_year, df_course


def get_admin_scatter_analytics(
    course: Optional[str] = None,
    year_of_study: Optional[int] = None,
    gender: Optional[str] = None,
    risk_category: Optional[str] = None,
) -> pd.DataFrame:
    """Returns joined tabular data of actual user assessments for dynamic scatter/distribution plots."""
    init_user_tables()
    conn, engine = get_connection()

    where_clauses = ["u.role = 'student'", "p.burnout_score IS NOT NULL"]
    params = []

    if course and course != "All":
        where_clauses.append("sp.course = ?" if engine == "sqlite" else "sp.course = %s")
        params.append(course)
    if year_of_study and year_of_study != "All":
        where_clauses.append("sp.year_of_study = ?" if engine == "sqlite" else "sp.year_of_study = %s")
        params.append(int(year_of_study))
    if gender and gender != "All":
        where_clauses.append("sp.gender = ?" if engine == "sqlite" else "sp.gender = %s")
        params.append(gender)
    if risk_category and risk_category != "All":
        where_clauses.append("p.risk_category = ?" if engine == "sqlite" else "p.risk_category = %s")
        params.append(risk_category)

    where_sql = f"WHERE {' AND '.join(where_clauses)}"

    query = f"""
    SELECT 
        u.user_id,
        u.name,
        sp.course,
        sp.year_of_study,
        sp.gender,
        sp.age,
        a.assessment_id,
        a.created_at AS assessment_date,
        a.study_hours_per_day,
        a.assignment_workload,
        a.exam_frequency,
        a.academic_pressure,
        a.sleep_hours,
        a.sleep_quality,
        a.screen_time_hours,
        a.physical_activity_hours,
        a.social_interaction_hours,
        a.breaks_per_day,
        a.hobbies_hours_per_week,
        a.days_off_per_week,
        a.attendance_percentage,
        a.cgpa,
        p.burnout_score,
        p.risk_category,
        p.academic_pressure_score,
        p.lifestyle_balance_score
    FROM users u
    JOIN student_profiles sp ON u.user_id = sp.user_id
    JOIN assessments a ON u.user_id = a.user_id
    JOIN predictions p ON a.assessment_id = p.assessment_id
    {where_sql}
    ORDER BY a.created_at DESC
    """

    df = pd.read_sql_query(query, conn, params=tuple(params) if params else None)
    conn.close()
    return df


def get_admin_student_records(
    search_query: Optional[str] = None,
    course: Optional[str] = None,
    year_of_study: Optional[int] = None,
    risk_category: Optional[str] = None,
) -> pd.DataFrame:
    """Returns privacy-conscious student assessment records for administrator review."""
    df = get_admin_scatter_analytics(course=course, year_of_study=year_of_study, risk_category=risk_category)
    if df.empty:
        return df

    if search_query and search_query.strip():
        q = search_query.strip().lower()
        df = df[df["name"].str.lower().str.contains(q) | df["course"].str.lower().str.contains(q)]

    return df


def get_admin_swot_analysis(
    course: Optional[str] = None,
    year_of_study: Optional[int] = None,
    gender: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Synthesizes data-driven institutional Strengths, Weaknesses, Opportunities,
    and Threats strictly derived from live database records.
    """
    kpis = get_admin_dashboard_kpis(course=course, year_of_study=year_of_study, gender=gender)
    total_assessments = kpis.get("total_assessments", 0)

    if total_assessments < 1:
        return {
            "available": False,
            "message": "Not enough submitted data to generate a reliable SWOT summary. As students complete assessments, aggregate SWOT insights will automatically be synthesized from database records.",
        }

    avg_burnout = kpis["avg_burnout_score"]
    avg_sleep = kpis["avg_sleep_hours"]
    avg_pressure = kpis["avg_academic_pressure"]
    avg_balance = kpis["avg_lifestyle_balance"]
    high_risk_pct = kpis["high_risk_percentage"]
    low_pct = round(kpis.get("low_count", 0) * 100.0 / total_assessments, 1) if total_assessments > 0 else 0.0

    strengths = []
    weaknesses = []
    opportunities = []
    threats = []

    # Dynamic Strengths
    if avg_balance >= 50.0:
        strengths.append(f"Healthy baseline lifestyle buffer with a cohort score of {avg_balance:.1f} / 100 across sleep and restorative habits.")
    if avg_sleep >= 6.5:
        strengths.append(f"Cohort average nightly sleep duration ({avg_sleep:.1f}h) meets baseline restorative thresholds.")
    if low_pct >= 35.0:
        strengths.append(f"{low_pct:.1f}% of student evaluations are positioned in the resilient Low Risk bracket.")
    if not strengths:
        strengths.append(f"Active engagement with {total_assessments} longitudinal burnout assessments logged in the platform.")

    # Dynamic Weaknesses
    if avg_sleep < 6.5:
        weaknesses.append(f"Sleep deprivation risk: cohort average sleep duration is restricted to {avg_sleep:.1f} hours per night.")
    if avg_pressure >= 50.0:
        weaknesses.append(f"Elevated academic strain index ({avg_pressure:.1f} / 100) driven by compounding workload and exam frequency.")
    if high_risk_pct >= 25.0:
        weaknesses.append(f"{high_risk_pct:.1f}% of student evaluations classify in the High or Very High risk categories.")
    if not weaknesses:
        weaknesses.append("Variance in daily rest break frequency across different academic years.")

    # Dynamic Opportunities
    opportunities.append("Institutional introduction of structured 10-minute micro-breaks to mitigate cognitive fatigue surges.")
    if avg_sleep < 6.5:
        opportunities.append("Campus digital wellness initiatives to reduce pre-sleep screen time and improve sleep quality.")
    opportunities.append("Workload pacing and assignment calendar coordination across academic departments.")

    # Dynamic Threats
    if high_risk_pct >= 30.0:
        threats.append(f"Concentrated burnout risk ({high_risk_pct:.1f}% in High/Very High) requires immediate institutional wellness intervention.")
    threats.append("Cumulative unbuffered exam pressure without designated recovery days accelerating student exhaustion.")

    return {
        "available": True,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "opportunities": opportunities,
        "threats": threats,
    }


# =====================================================================
# DEFAULT ACCOUNTS & ADMIN INITIALIZATION
# =====================================================================

def ensure_admin_account():
    """
    Ensures that the default administrator account exists in the database:
    - Admin: admin@mindmap.ai / AdminPass2026! (role: admin)
    
    Zero fake student accounts, zero synthetic assessments, and zero fake predictions
    are created. Live student records and statistics are strictly populated by real users.
    """
    init_user_tables()
    from utils.auth import hash_password

    admin_user = get_user_by_email("admin@mindmap.ai")
    if not admin_user:
        pwd_hash, salt = hash_password("AdminPass2026!")
        create_user(
            name="System Administrator",
            username="admin",
            email="admin@mindmap.ai",
            password_hash=pwd_hash,
            salt=salt,
            role="admin",
        )


def seed_default_accounts():
    """Alias for ensure_admin_account for backwards compatibility."""
    ensure_admin_account()


def seed_demo_user():
    """Alias for ensure_admin_account for backwards compatibility."""
    ensure_admin_account()

