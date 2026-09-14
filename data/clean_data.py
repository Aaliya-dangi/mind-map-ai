"""
MindMap AI - Data Cleaning Module
Implements data validation, de-duplication, imputation, and type coercion
per design.md §15 and RMD.md §1.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

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


def clean_student_dataset(
    raw_file_path: Path = config.RAW_DATA_FILE,
    output_file_path: Path = config.PROCESSED_DATA_FILE,
    summary_file_path: Path = config.PROCESSED_DATA_DIR / "cleaning_summary.json",
) -> pd.DataFrame:
    """
    Cleans raw student dataset following the Data Cleaning Strategy (design.md §15).

    Cleaning Pipeline:
    1. Ingestion & initial auditing
    2. De-duplication on student_id
    3. Missing value handling (median for numeric, mode for categorical)
    4. Range boundary enforcement (clipping out-of-range synthetic noise)
    5. Derived feature re-synchronization
    6. Type coercion & precision normalization
    7. Persistence and audit logging
    """
    print(f"Loading raw dataset from {raw_file_path}...")
    df = pd.read_csv(raw_file_path)
    initial_rows = len(df)
    initial_cols = len(df.columns)

    cleaning_log = {
        "initial_row_count": initial_rows,
        "initial_column_count": initial_cols,
        "duplicates_removed": 0,
        "imputations": {},
        "range_adjustments": {},
    }

    # Step 1: De-duplication on student_id
    duplicates = df.duplicated(subset=["student_id"], keep="first")
    num_dups = int(duplicates.sum())
    if num_dups > 0:
        df = df.drop_duplicates(subset=["student_id"], keep="first").reset_index(drop=True)
        cleaning_log["duplicates_removed"] = num_dups
        print(f"-> Removed {num_dups} duplicate records on 'student_id'.")

    # Step 2: Handle Missing Values
    null_counts = df.isnull().sum()
    for col in df.columns:
        null_count = int(null_counts[col])
        if null_count > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                impute_val = float(df[col].median())
                df[col] = df[col].fillna(impute_val)
                cleaning_log["imputations"][col] = {
                    "missing_count": null_count,
                    "strategy": "median",
                    "value": round(impute_val, 2),
                }
                print(f"-> Imputed {null_count} missing values in '{col}' with median ({impute_val:.2f}).")
            else:
                mode_val = str(df[col].mode()[0])
                df[col] = df[col].fillna(mode_val)
                cleaning_log["imputations"][col] = {
                    "missing_count": null_count,
                    "strategy": "mode",
                    "value": mode_val,
                }
                print(f"-> Imputed {null_count} missing values in '{col}' with mode ('{mode_val}').")

    # Step 3: Range Validations & Clipping (per Data Dictionary §14)
    feature_bounds = {
        "age": (18, 28, int),
        "year_of_study": (1, 4, int),
        "attendance_percentage": (40.0, 100.0, float),
        "cgpa": (4.0, 10.0, float),
        "study_hours_per_day": (0.0, 12.0, float),
        "assignment_workload": (1, 10, int),
        "exam_frequency": (0, 5, int),
        "academic_pressure": (1, 10, int),
        "sleep_hours": (3.0, 10.0, float),
        "sleep_quality": (1, 10, int),
        "screen_time_hours": (1.0, 14.0, float),
        "physical_activity_hours": (0.0, 10.0, float),
        "social_interaction_hours": (0.0, 20.0, float),
        "breaks_per_day": (0, 8, int),
        "hobbies_hours_per_week": (0.0, 15.0, float),
        "days_off_per_week": (0, 3, int),
    }

    for col, (min_v, max_v, dtype_cast) in feature_bounds.items():
        if col in df.columns:
            out_of_bounds = (df[col] < min_v) | (df[col] > max_v)
            count_oob = int(out_of_bounds.sum())
            if count_oob > 0:
                cleaning_log["range_adjustments"][col] = {
                    "out_of_bounds_count": count_oob,
                    "bounds": [min_v, max_v],
                }
            df[col] = np.clip(df[col], min_v, max_v)

    # Step 4: Recompute Composite Features for 100% Integrity
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

    # Ensure burnout_score is bounded and categorical target is consistent
    df["burnout_score"] = np.round(np.clip(df["burnout_score"], 0.0, 100.0), 2)
    df["burnout_risk"] = assign_risk_category(df["burnout_score"])

    # Step 5: Enforce Explicit Data Types
    type_dict = {
        "student_id": "string",
        "age": "int64",
        "gender": "category",
        "year_of_study": "int64",
        "course": "category",
        "assignment_workload": "int64",
        "exam_frequency": "int64",
        "academic_pressure": "int64",
        "sleep_quality": "int64",
        "breaks_per_day": "int64",
        "days_off_per_week": "int64",
        "burnout_risk": "category",
    }
    for col, dt in type_dict.items():
        if col in df.columns:
            if dt == "category":
                df[col] = df[col].astype(str)
            else:
                df[col] = df[col].astype(dt)

    # Round floating columns to 2 decimal places
    float_cols = df.select_dtypes(include=["float64", "float32"]).columns
    for c in float_cols:
        df[c] = df[c].round(2)

    # Final audit log
    cleaning_log["final_row_count"] = len(df)
    cleaning_log["final_column_count"] = len(df.columns)
    cleaning_log["risk_distribution"] = df["burnout_risk"].value_counts().to_dict()

    # Step 6: Save cleaned dataset & audit summary
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file_path, index=False)
    print(f"[OK] Cleaned dataset saved to {output_file_path}")
    print(f"     Rows: {len(df)}, Columns: {len(df.columns)}")

    with open(summary_file_path, "w", encoding="utf-8") as f:
        json.dump(cleaning_log, f, indent=2)
    print(f"[OK] Cleaning audit log saved to {summary_file_path}")

    return df


def main():
    clean_student_dataset()


if __name__ == "__main__":
    main()
