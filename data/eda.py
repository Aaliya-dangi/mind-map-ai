"""
MindMap AI - Exploratory Data Analysis & Statistical Analysis
Computes univariate statistics, bivariate correlations, cohort aggregations,
and group hypothesis tests per design.md §16-17.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

# Add project root to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config


def run_exploratory_data_analysis(
    dataset_path: Path = config.PROCESSED_DATA_FILE,
    output_summary_path: Path = config.PROCESSED_DATA_DIR / "eda_summary.json",
) -> dict:
    """
    Performs comprehensive statistical EDA on the cleaned student dataset.

    Returns:
    - Dict containing univariate stats, bivariate correlations, cohort comparisons,
      and statistical test results.
    """
    print(f"Loading cleaned dataset from {dataset_path} for EDA...")
    df = pd.read_csv(dataset_path)

    # 1. Numerical & Categorical Feature Columns
    numerical_cols = [
        "age",
        "attendance_percentage",
        "cgpa",
        "study_hours_per_day",
        "assignment_workload",
        "exam_frequency",
        "academic_pressure",
        "sleep_hours",
        "sleep_quality",
        "screen_time_hours",
        "physical_activity_hours",
        "social_interaction_hours",
        "breaks_per_day",
        "hobbies_hours_per_week",
        "days_off_per_week",
        "academic_pressure_score",
        "lifestyle_balance_score",
        "burnout_score",
    ]

    # 2. Univariate Numerical Statistics
    univariate_stats = {}
    for col in numerical_cols:
        series = df[col]
        univariate_stats[col] = {
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2),
            "median": round(float(series.median()), 2),
            "min": round(float(series.min()), 2),
            "q25": round(float(series.quantile(0.25)), 2),
            "q75": round(float(series.quantile(0.75)), 2),
            "max": round(float(series.max()), 2),
            "skewness": round(float(series.skew()), 3),
        }

    # 3. Bivariate Correlations with burnout_score
    bivariate_correlations = {}
    for col in numerical_cols:
        if col == "burnout_score":
            continue
        pearson_r, pearson_p = stats.pearsonr(df[col], df["burnout_score"])
        spearman_r, spearman_p = stats.spearmanr(df[col], df["burnout_score"])
        bivariate_correlations[col] = {
            "pearson_r": round(float(pearson_r), 3),
            "pearson_p": float(f"{pearson_p:.2e}"),
            "spearman_rho": round(float(spearman_r), 3),
            "spearman_p": float(f"{spearman_p:.2e}"),
            "direction": "positive" if pearson_r > 0 else "negative",
        }

    # Sort correlations by absolute Pearson r
    ranked_correlations = sorted(
        bivariate_correlations.items(),
        key=lambda item: abs(item[1]["pearson_r"]),
        reverse=True,
    )

    # 4. Cohort Comparisons by Course
    course_cohort = {}
    for course, group in df.groupby("course", observed=False):
        course_cohort[str(course)] = {
            "student_count": len(group),
            "avg_burnout_score": round(float(group["burnout_score"].mean()), 2),
            "avg_academic_pressure": round(float(group["academic_pressure_score"].mean()), 2),
            "avg_lifestyle_balance": round(float(group["lifestyle_balance_score"].mean()), 2),
            "avg_sleep_hours": round(float(group["sleep_hours"].mean()), 2),
            "high_risk_percentage": round(float((group["burnout_risk"].isin(["High", "Very High"])).mean() * 100), 2),
        }

    # 5. Cohort Comparisons by Year of Study
    year_cohort = {}
    for yr, group in df.groupby("year_of_study", observed=False):
        year_cohort[str(yr)] = {
            "student_count": len(group),
            "avg_burnout_score": round(float(group["burnout_score"].mean()), 2),
            "avg_academic_pressure": round(float(group["academic_pressure_score"].mean()), 2),
            "avg_lifestyle_balance": round(float(group["lifestyle_balance_score"].mean()), 2),
            "avg_sleep_hours": round(float(group["sleep_hours"].mean()), 2),
            "high_risk_percentage": round(float((group["burnout_risk"].isin(["High", "Very High"])).mean() * 100), 2),
        }

    # 6. Statistical Significance Tests (ANOVA across Cohorts)
    # Test 1: Burnout differences across Courses
    course_groups = [group["burnout_score"].values for _, group in df.groupby("course", observed=False)]
    f_course, p_course = stats.f_oneway(*course_groups)

    # Test 2: Burnout differences across Years of Study
    year_groups = [group["burnout_score"].values for _, group in df.groupby("year_of_study", observed=False)]
    f_year, p_year = stats.f_oneway(*year_groups)

    anova_results = {
        "course_differences": {
            "f_statistic": round(float(f_course), 3),
            "p_value": float(f"{p_course:.2e}"),
            "statistically_significant": bool(p_course < 0.05),
        },
        "year_of_study_differences": {
            "f_statistic": round(float(f_year), 3),
            "p_value": float(f"{p_year:.2e}"),
            "statistically_significant": bool(p_year < 0.05),
        },
    }

    # 7. Assemble Structured Summary
    eda_summary = {
        "dataset_metadata": {
            "total_students": len(df),
            "total_features": len(df.columns),
            "risk_distribution": df["burnout_risk"].value_counts().to_dict(),
        },
        "univariate_stats": univariate_stats,
        "bivariate_correlations": dict(ranked_correlations),
        "cohort_by_course": course_cohort,
        "cohort_by_year": year_cohort,
        "anova_hypothesis_tests": anova_results,
    }

    # Save to file
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_summary_path, "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)

    print(f"[OK] EDA summary exported to {output_summary_path}")
    print("\n--- Key Correlation Drivers with Burnout Score ---")
    for feat, corr_info in ranked_correlations[:8]:
        print(f"  {feat:30s} | Pearson r = {corr_info['pearson_r']:+.3f} (p = {corr_info['pearson_p']})")

    print("\n--- Cohort Risk Highlights by Course ---")
    for course, metrics in course_cohort.items():
        print(f"  {course:20s} | Avg Burnout: {metrics['avg_burnout_score']:.1f} | High/Very High Risk: {metrics['high_risk_percentage']:.1f}%")

    return eda_summary


def main():
    run_exploratory_data_analysis()


if __name__ == "__main__":
    main()
