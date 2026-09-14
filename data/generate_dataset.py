"""
MindMap AI - Synthetic Dataset Generator
Generates realistic, non-deterministic student burnout dataset matching the
Data Dictionary in design.md §14 and requirements in RMD.md §5.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from features.engineering import (
    calculate_academic_pressure_score,
    calculate_lifestyle_balance_score,
    assign_risk_category,
)


def generate_student_dataset(
    num_records: int = 3000,
    random_state: int = 42,
    introduce_raw_artifacts: bool = True,
) -> pd.DataFrame:
    """
    Generates synthetic student records with realistic inter-feature correlations
    and non-deterministic target variance (noise).

    Parameters:
    - num_records: Number of student records to generate (default 3000)
    - random_state: Seed for reproducibility
    - introduce_raw_artifacts: Whether to add realistic missing values / noise for raw CSV

    Returns:
    - pd.DataFrame containing all 22 Data Dictionary fields
    """
    np.random.seed(random_state)

    # 1. Identifiers & Demographics
    student_ids = [f"STU{i+1:04d}" for i in range(num_records)]
    ages = np.random.choice(
        [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
        size=num_records,
        p=[0.08, 0.22, 0.25, 0.22, 0.11, 0.05, 0.03, 0.02, 0.01, 0.005, 0.005],
    )
    genders = np.random.choice(["Female", "Male", "Other"], size=num_records, p=[0.49, 0.47, 0.04])
    years_of_study = np.random.choice([1, 2, 3, 4], size=num_records, p=[0.27, 0.26, 0.25, 0.22])

    courses = np.random.choice(
        ["Computer Science", "Engineering", "Medicine", "Commerce", "Design", "Business", "Arts"],
        size=num_records,
        p=[0.24, 0.20, 0.12, 0.16, 0.10, 0.10, 0.08],
    )

    # Latent stress profile (heterogeneity across students: 0 = relaxed/balanced, 1 = high stress)
    latent_stress = np.random.beta(a=2.0, b=2.2, size=num_records)  # spread across 0 to 1

    # Course-specific workload modifiers
    course_workload_bias = {
        "Medicine": 1.35,
        "Engineering": 1.20,
        "Computer Science": 1.10,
        "Commerce": 0.95,
        "Business": 0.90,
        "Design": 1.00,
        "Arts": 0.80,
    }
    course_screen_bias = {
        "Computer Science": 2.2,
        "Design": 1.8,
        "Engineering": 1.2,
        "Business": 0.8,
        "Commerce": 0.6,
        "Medicine": 0.4,
        "Arts": 0.2,
    }

    # 2. Academic Factors
    study_hours = []
    assignment_workload = []
    exam_frequency = []
    academic_pressure = []
    attendance = []
    cgpa = []

    for i in range(num_records):
        c = courses[i]
        yr = years_of_study[i]
        ls = latent_stress[i]
        bias = course_workload_bias[c]

        # Base study hours: 1.0 to 11.0 + course bias + latent stress + senior year effect
        base_study = (2.2 + 5.5 * ls) * bias + (0.35 * yr) + np.random.normal(0, 0.9)
        study_h = float(np.clip(round(base_study, 1), 0.5, 12.0))
        study_hours.append(study_h)

        # Workload scale 1-10
        base_workload = int(np.clip(round(1.5 + 7.0 * ls * bias + (0.3 * yr) + np.random.normal(0, 1.1)), 1, 10))
        assignment_workload.append(base_workload)

        # Monthly exam frequency 0-5
        base_exams = int(np.clip(np.random.poisson(lam=0.8 + 2.4 * ls * bias), 0, 5))
        exam_frequency.append(base_exams)

        # Attendance 40-100% (stressed students might have slightly lower attendance)
        base_att = 92.0 - (18.0 * ls) - (1.2 * yr) + np.random.normal(0, 7.5)
        att = float(np.clip(round(base_att, 1), 40.0, 100.0))
        attendance.append(att)

        # CGPA 4.0 - 10.0 (correlated with study & attendance + moderate noise)
        base_cgpa = 6.0 + (study_h * 0.22) + ((att - 60) * 0.025) - (ls * 0.6) + np.random.normal(0, 0.65)
        cgpa_val = float(np.clip(round(base_cgpa, 2), 4.0, 10.0))
        cgpa.append(cgpa_val)

        # Academic pressure scale 1-10
        pressure_signal = 1.0 + 8.0 * ls + (0.15 * (study_h - 4)) + np.random.normal(0, 0.8)
        acad_press = int(np.clip(round(pressure_signal), 1, 10))
        academic_pressure.append(acad_press)

    # 3. Lifestyle Factors
    sleep_hours = []
    sleep_quality = []
    screen_time = []
    phys_act = []
    social_int = []
    breaks = []
    hobbies = []
    days_off = []

    for i in range(num_records):
        c = courses[i]
        study_h = study_hours[i]
        ls = latent_stress[i]
        screen_bias = course_screen_bias[c]

        # Screen time 1-14 hours
        base_screen = 3.5 + screen_bias + (4.5 * ls) + (study_h * 0.15) + np.random.normal(0, 1.3)
        screen_h = float(np.clip(round(base_screen, 1), 1.0, 14.0))
        screen_time.append(screen_h)

        # Sleep hours 3-10 (depressed by stress, high study, high screen)
        base_sleep = 9.2 - (4.0 * ls) - (screen_h * 0.08) + np.random.normal(0, 0.75)
        sleep_h = float(np.clip(round(base_sleep, 1), 3.0, 10.0))
        sleep_hours.append(sleep_h)

        # Sleep quality scale 1-10
        base_sq = 9.5 - (7.0 * ls) - (screen_h * 0.1) + (sleep_h * 0.2) + np.random.normal(0, 0.9)
        sq_val = int(np.clip(round(base_sq), 1, 10))
        sleep_quality.append(sq_val)

        # Physical activity hours per week 0-10
        base_phys = 7.5 - (6.0 * ls) + np.random.normal(0, 1.4)
        phys_val = max(0.0, float(np.clip(round(base_phys, 1), 0.0, 10.0)))
        phys_act.append(phys_val)

        # Social interaction hours per week 0-20
        base_soc = 16.0 - (12.0 * ls) + np.random.normal(0, 2.5)
        soc_val = max(0.0, float(np.clip(round(base_soc, 1), 0.0, 20.0)))
        social_int.append(soc_val)

        # Breaks per day 0-8
        base_brk = int(np.clip(round(6.5 - (5.0 * ls) + np.random.normal(0, 0.9)), 0, 8))
        breaks.append(base_brk)

        # Hobbies hours per week 0-15
        base_hob = 11.0 - (9.0 * ls) + np.random.normal(0, 1.8)
        hob_val = max(0.0, float(np.clip(round(base_hob, 1), 0.0, 15.0)))
        hobbies.append(hob_val)

        # Days off per week 0-3
        base_doff = int(np.clip(round(2.5 - (2.0 * ls) + np.random.normal(0, 0.5)), 0, 3))
        days_off.append(base_doff)

    # Assemble Base DataFrame
    df = pd.DataFrame({
        "student_id": student_ids,
        "age": ages,
        "gender": genders,
        "year_of_study": years_of_study,
        "course": courses,
        "attendance_percentage": attendance,
        "cgpa": cgpa,
        "study_hours_per_day": study_hours,
        "assignment_workload": assignment_workload,
        "exam_frequency": exam_frequency,
        "academic_pressure": academic_pressure,
        "sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality,
        "screen_time_hours": screen_time,
        "physical_activity_hours": phys_act,
        "social_interaction_hours": social_int,
        "breaks_per_day": breaks,
        "hobbies_hours_per_week": hobbies,
        "days_off_per_week": days_off,
    })

    # 4. Compute Derived Composite Indices
    df["academic_pressure_score"] = calculate_academic_pressure_score(
        academic_pressure=df["academic_pressure"],
        assignment_workload=df["assignment_workload"],
        exam_frequency=df["exam_frequency"],
        attendance_percentage=df["attendance_percentage"],
        cgpa=df["cgpa"],
        study_hours_per_day=df["study_hours_per_day"],
    )

    df["lifestyle_balance_score"] = calculate_lifestyle_balance_score(
        sleep_hours=df["sleep_hours"],
        sleep_quality=df["sleep_quality"],
        screen_time_hours=df["screen_time_hours"],
        physical_activity_hours=df["physical_activity_hours"],
        social_interaction_hours=df["social_interaction_hours"],
        breaks_per_day=df["breaks_per_day"],
        hobbies_hours_per_week=df["hobbies_hours_per_week"],
        days_off_per_week=df["days_off_per_week"],
    )

    # 5. Compute Burnout Score with Realistic Gaussian Noise
    # Combination of pressure score, lifestyle deficit, and non-deterministic Gaussian noise
    lifestyle_deficit = 100.0 - df["lifestyle_balance_score"]
    base_signal = 0.50 * df["academic_pressure_score"] + 0.50 * lifestyle_deficit
    noise = np.random.normal(loc=0.0, scale=7.5, size=num_records)
    raw_burnout_score = np.clip(base_signal + noise, 0.0, 100.0)
    df["burnout_score"] = np.round(raw_burnout_score, 2)

    # 6. Assign Categorical Burnout Risk Target
    df["burnout_risk"] = assign_risk_category(df["burnout_score"])

    # 7. Optionally introduce subtle raw artifacts for the raw export
    # (e.g. realistic data anomalies for Phase 3 cleaning to validate)
    if introduce_raw_artifacts:
        raw_df = df.copy()
        # Introduce a few missing values (~1.5% in optional lifestyle fields)
        nan_indices_hob = np.random.choice(raw_df.index, size=int(0.015 * num_records), replace=False)
        raw_df.loc[nan_indices_hob, "hobbies_hours_per_week"] = np.nan
        nan_indices_sq = np.random.choice(raw_df.index, size=int(0.012 * num_records), replace=False)
        raw_df.loc[nan_indices_sq, "sleep_quality"] = np.nan
        
        # Add a tiny handful of duplicate rows (5 duplicate entries)
        dup_indices = np.random.choice(raw_df.index, size=5, replace=False)
        duplicates = raw_df.loc[dup_indices].copy()
        raw_df = pd.concat([raw_df, duplicates], ignore_index=True)
        return raw_df

    return df


def main():
    """Generates and saves the raw synthetic dataset."""
    print("Generating synthetic student burnout dataset...")
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = generate_student_dataset(num_records=3000, random_state=42, introduce_raw_artifacts=True)
    df_raw.to_csv(config.RAW_DATA_FILE, index=False)
    print(f"[OK] Raw dataset saved to {config.RAW_DATA_FILE}")
    print(f"     Total rows: {len(df_raw)}, Columns: {len(df_raw.columns)}")
    print(f"     Columns: {list(df_raw.columns)}")
    print(f"\nRisk Distribution:\n{df_raw['burnout_risk'].value_counts(dropna=False)}")


if __name__ == "__main__":
    main()
