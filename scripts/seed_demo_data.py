"""
MindMap AI - Optional Standalone Demo Data Seeder
Run this script manually ONLY if you want to populate mock student data for local testing.
This script is NEVER executed automatically by the application.

Usage:
    python scripts/seed_demo_data.py
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from db.user_db import create_user, get_user_by_email, save_user_assessment_and_prediction
from utils.auth import hash_password
from ml.predict import predict_burnout_risk
from utils.recommendations import generate_data_driven_recommendations


def seed_demo_data():
    print("Seeding sample cohort for local testing...")
    sample_students = [
        {"name": "Rohan Sharma", "email": "rohan.sharma@campus.edu", "age": 22, "gender": "Male", "year": 4, "course": "Medicine", "study": 9.0, "sleep": 4.5, "sq": 3, "screen": 10.0, "workload": 9, "pressure": 9, "breaks": 1, "phys": 1.0, "hobbies": 1.0, "days_off": 0},
        {"name": "Priya Patel", "email": "priya.patel@campus.edu", "age": 20, "gender": "Female", "year": 2, "course": "Computer Science", "study": 7.5, "sleep": 5.5, "sq": 5, "screen": 9.0, "workload": 7, "pressure": 7, "breaks": 2, "phys": 2.5, "hobbies": 3.0, "days_off": 1},
        {"name": "Aarav Gupta", "email": "aarav.gupta@campus.edu", "age": 19, "gender": "Male", "year": 1, "course": "Business", "study": 4.0, "sleep": 7.5, "sq": 8, "screen": 6.0, "workload": 4, "pressure": 4, "breaks": 4, "phys": 5.0, "hobbies": 6.0, "days_off": 2},
        {"name": "Sneha Verma", "email": "sneha.verma@campus.edu", "age": 21, "gender": "Female", "year": 3, "course": "Design", "study": 5.0, "sleep": 6.8, "sq": 7, "screen": 7.0, "workload": 5, "pressure": 5, "breaks": 3, "phys": 4.0, "hobbies": 8.0, "days_off": 1},
        {"name": "Vikram Singh", "email": "vikram.singh@campus.edu", "age": 23, "gender": "Male", "year": 4, "course": "Engineering", "study": 8.5, "sleep": 5.0, "sq": 4, "screen": 9.5, "workload": 8, "pressure": 8, "breaks": 2, "phys": 1.5, "hobbies": 2.0, "days_off": 0},
        {"name": "Ananya Joshi", "email": "ananya.joshi@campus.edu", "age": 20, "gender": "Female", "year": 2, "course": "Arts", "study": 3.5, "sleep": 8.2, "sq": 9, "screen": 5.0, "workload": 3, "pressure": 3, "breaks": 5, "phys": 6.0, "hobbies": 10.0, "days_off": 2},
        {"name": "Karan Mehta", "email": "karan.mehta@campus.edu", "age": 22, "gender": "Male", "year": 3, "course": "Commerce", "study": 6.0, "sleep": 6.2, "sq": 6, "screen": 7.5, "workload": 6, "pressure": 6, "breaks": 3, "phys": 3.5, "hobbies": 4.0, "days_off": 1},
        {"name": "Meera Nair", "email": "meera.nair@campus.edu", "age": 21, "gender": "Female", "year": 3, "course": "Medicine", "study": 9.5, "sleep": 4.8, "sq": 4, "screen": 8.5, "workload": 9, "pressure": 9, "breaks": 1, "phys": 1.0, "hobbies": 1.5, "days_off": 0},
    ]

    for s in sample_students:
        if not get_user_by_email(s["email"]):
            pwd_hash, salt = hash_password("StudentPass2026!")
            u = create_user(
                name=s["name"],
                email=s["email"],
                password_hash=pwd_hash,
                salt=salt,
                role="student",
                age=s["age"],
                gender=s["gender"],
                year_of_study=s["year"],
                course=s["course"],
            )
            if u:
                payload = {
                    "age": s["age"],
                    "gender": s["gender"],
                    "year_of_study": s["year"],
                    "course": s["course"],
                    "attendance_percentage": 82.0,
                    "cgpa": 7.8,
                    "study_hours_per_day": s["study"],
                    "assignment_workload": s["workload"],
                    "exam_frequency": 2,
                    "academic_pressure": s["pressure"],
                    "sleep_hours": s["sleep"],
                    "sleep_quality": s["sq"],
                    "screen_time_hours": s["screen"],
                    "physical_activity_hours": s["phys"],
                    "social_interaction_hours": 7.0,
                    "breaks_per_day": s["breaks"],
                    "hobbies_hours_per_week": s["hobbies"],
                    "days_off_per_week": s["days_off"],
                }
                pred = predict_burnout_risk(payload)
                recs = generate_data_driven_recommendations(payload, pred)
                save_user_assessment_and_prediction(u["user_id"], payload, pred, recs)
                print(f"Created student record for {s['name']} ({s['course']})")

    print("[OK] Demo data seeding completed.")


if __name__ == "__main__":
    seed_demo_data()
