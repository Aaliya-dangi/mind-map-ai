# MindMap AI — Design Document
## Student Burnout Analytics & Predictive Risk Assessment Platform

**Tagline:** "Understand the Pattern. Predict the Risk. Take Back Control."

---

## 1. Project Overview

MindMap AI is a data science portfolio product that analyzes academic and lifestyle patterns among students to surface burnout-risk signals. It combines SQL-backed analytics, exploratory data analysis, feature engineering, and a machine learning prediction pipeline, delivered through an interactive Streamlit dashboard.

The project is designed to demonstrate end-to-end data science capability — from raw data to a usable analytics product — rather than a single notebook exercise.

## 2. Vision

Build a small, credible analytics product that a student, educator, or wellness office could plausibly use to understand aggregate and individual burnout-risk patterns, backed by transparent, explainable modeling rather than a black box.

## 3. Problem Statement

Students face compounding academic and lifestyle pressures (workload, sleep loss, screen time, reduced activity) that are rarely tracked or surfaced until burnout is already severe. There is no lightweight, data-driven way to spot patterns early, compare a given student's situation to peers, or simulate the effect of lifestyle changes.

## 4. Target Users

- Individual students exploring their own risk patterns
- Academic advisors / student-wellness staff reviewing cohort-level trends
- Recruiters/reviewers evaluating this as a data science portfolio artifact

## 5. Goals

- Demonstrate a full pipeline: cleaning → SQL → EDA → feature engineering → ML → explainability → dashboard
- Provide an interactive "What-If Simulator" for lifestyle changes
- Present risk transparently, with explainable contributing factors
- Ship a working MVP within ~2 days

## 6. Non-Goals

- Medical diagnosis of burnout, depression, or anxiety
- Clinical-grade prediction or treatment recommendations
- Real-time data collection from real students
- Deep learning, LLM APIs, chatbots, authentication systems, or cloud infrastructure

## 7. Key Features

1. Overview Dashboard with KPI cards
2. Student Burnout Analytics (cohort-level)
3. Individual Burnout Risk Prediction
4. Burnout Risk Score (0–100) and category (Low/Moderate/High/Very High)
5. Academic Pressure Score (derived, 0–100)
6. Lifestyle Balance Score (derived, 0–100)
7. Explainable prediction (top contributing factors)
8. Feature importance visualization
9. Interactive data exploration with filters/drill-down
10. SQL-backed analytics queries
11. What-If Burnout Simulator
12. Data-driven, non-medical recommendations
13. ML model comparison (Logistic Regression vs Random Forest)
14. Model performance analysis (Accuracy, Precision, Recall, F1, Confusion Matrix)
15. Optional downloadable report

## 8. User Personas

- **Aditi, final-year student**: curious about her own burnout risk and what she could change.
- **Prof. Rao, academic advisor**: wants a cohort view — which year/course shows the highest risk.
- **Hiring manager reviewing portfolio**: wants to see technical range (SQL, stats, ML, viz, product thinking) in one place.

## 9. User Journey

1. Land on Overview → see cohort KPIs and top-line risk distribution.
2. Explore Analytics → filter by year/course, inspect relationships (sleep vs burnout, etc.).
3. Go to "My Risk" → enter personal academic/lifestyle inputs → get a risk score, category, and top contributing factors.
4. Open What-If Simulator → adjust sliders (sleep, study hours, screen time) → see recalculated risk and the delta.
5. Review Model Insights → understand how the model was built/evaluated and its limitations.

## 10. Application Flow

```
Student Data → Data Cleaning → MySQL → EDA → Feature Engineering →
Statistical Analysis → Interactive Visualization → ML Training/Comparison →
Explainable Prediction → Streamlit Dashboard → What-If Simulation →
Data-Driven Recommendations
```

## 11. System Architecture

- **Data layer**: CSV/synthetic dataset generated once, loaded into MySQL tables (`students`, `academic_data`, `lifestyle_data`, `burnout_predictions`).
- **Processing layer**: Python scripts/notebooks for cleaning, feature engineering, and model training; trained models persisted via Joblib.
- **Analytics layer**: SQL queries (via a MySQL connector) feeding cohort-level aggregates to the dashboard.
- **App layer**: Streamlit multipage app that reads from MySQL (aggregates) and from the persisted model (prediction/simulation), rendering Plotly visualizations.

```
[MySQL DB] <---- SQL queries ---- [Streamlit App] ----> [Joblib Model] ----> [Prediction + Explainability]
     ^                                    |
     |                                    v
[Cleaned Dataset]                 [Plotly Visualizations]
```

## 12. Technology Stack

Python, Pandas, NumPy, Scikit-learn, MySQL, SQL, Plotly, Streamlit, Joblib. Matplotlib/Seaborn optionally for static EDA plots during development. No deep learning, LLM APIs, paid APIs, chatbots, auth systems, microservices, or cloud infra.

## 13. Dataset Design

Synthetic dataset of individual student records. See Data Dictionary below. Relationships between lifestyle/academic variables and `burnout_score` are built in with realistic noise — not a deterministic formula — so the ML model reflects genuine (imperfect) predictive signal rather than a trivially solvable target.

## 14. Data Dictionary

| Feature | Type | Range/Values | Categorical/Numerical | Raw/Derived | Purpose |
|---|---|---|---|---|---|
| student_id | string | unique ID | — | raw | Row identifier |
| age | int | 18–28 | numerical | raw | Demographic context |
| gender | categorical | Male/Female/Other | categorical | raw | Demographic context |
| year_of_study | categorical | 1–4 | categorical | raw | Cohort comparison |
| course | categorical | e.g. CS, Commerce, Design | categorical | raw | Cohort comparison |
| attendance_percentage | float | 40–100 | numerical | raw | Engagement proxy |
| cgpa | float | 4.0–10.0 | numerical | raw | Academic performance |
| study_hours_per_day | float | 0–12 | numerical | raw | Workload |
| assignment_workload | int (scale) | 1–10 | numerical | raw | Workload |
| exam_frequency | int | 0–5 per month | numerical | raw | Academic pressure input |
| academic_pressure | int (scale) | 1–10 | numerical | raw | Self-reported pressure |
| sleep_hours | float | 3–10 | numerical | raw | Lifestyle input |
| sleep_quality | int (scale) | 1–10 | numerical | raw | Lifestyle input |
| screen_time_hours | float | 1–14 | numerical | raw | Lifestyle input |
| physical_activity_hours | float | 0–10 per week | numerical | raw | Lifestyle input |
| social_interaction_hours | float | 0–20 per week | numerical | raw | Lifestyle input |
| breaks_per_day | int | 0–8 | numerical | raw | Lifestyle input |
| hobbies_hours_per_week | float | 0–15 | numerical | raw | Lifestyle input |
| days_off_per_week | int | 0–3 | numerical | raw | Lifestyle input |
| academic_pressure_score | float | 0–100 | numerical | derived | Composite feature |
| lifestyle_balance_score | float | 0–100 | numerical | derived | Composite feature |
| burnout_score | float | 0–100 | numerical | derived (target-adjacent) | Continuous risk measure |
| burnout_risk | categorical | Low/Moderate/High/Very High | categorical | derived (label) | Classification target |

## 15. Data Cleaning Strategy

- Validate ranges per field; clip or flag out-of-range synthetic noise.
- Handle missing values: numerical → median imputation (documented), categorical → mode or "Unknown" bucket.
- De-duplicate on `student_id`.
- Type coercion and consistent categorical encoding vocab.
- Document all cleaning steps and row-count impact.

## 16. EDA Strategy

- Univariate distributions for all numerical features.
- Bivariate analysis: each lifestyle/academic variable vs `burnout_score`.
- Correlation heatmap across numerical features.
- Group comparisons: burnout by year_of_study, by course.
- Outlier inspection (boxplots) before feature engineering.

## 17. Statistical Analysis

- Correlation coefficients (Pearson/Spearman) between key predictors and burnout_score.
- Group-mean comparisons (e.g., ANOVA or simple t-tests) across year_of_study/course where sample size allows.
- Basic descriptive statistics (mean, median, std) surfaced in the Overview dashboard.

## 18. Feature Engineering

- **academic_pressure_score**: normalized 0–100 composite of academic_pressure, assignment_workload, exam_frequency, attendance_percentage (inverse), cgpa (inverse).
- **lifestyle_balance_score**: normalized 0–100 composite of sleep_hours, sleep_quality, physical_activity_hours, breaks_per_day, hobbies_hours_per_week, social_interaction_hours, days_off_per_week.
- Scaling (StandardScaler/MinMaxScaler) applied consistently at train and inference time.
- Categorical encoding (one-hot or ordinal, documented per feature).

## 19. ML Pipeline

Preprocessing → missing-value handling → encoding → scaling → train/test split (stratified) → feature engineering integration → model training → evaluation → model comparison → persistence (Joblib) → prediction → explainability.

## 20. Model Comparison

- Logistic Regression (baseline, interpretable coefficients)
- Random Forest (non-linear, feature importances)
- Optional third simple model (e.g., Gradient Boosting) if time allows

## 21. Model Evaluation

Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix — reported per model, per class, displayed in "Model Insights" page. No artificial accuracy inflation; realistic noise in the dataset keeps accuracy in a believable range (not 99–100%).

## 22. Explainability Strategy

- Random Forest feature importances (and/or permutation importance) surfaced per prediction.
- Logistic Regression coefficients shown as directionally interpretable.
- Per-prediction "top contributing factors" list, phrased as influence, not causation: *"These features had the strongest influence on the model's prediction."*

## 23. Burnout Risk Scoring

- Model outputs a probability; probability × 100 → **Burnout Risk Score (0–100)**.
- Categories (project-defined, not medical): 0–30 Low, 31–60 Moderate, 61–80 High, 81–100 Very High.

## 24. Dashboard Architecture

Streamlit multipage app:
- Overview
- Analytics
- My Risk
- What-If Simulator
- Risk Factors
- Model Insights

Shared state: cleaned dataset (cached), trained model + scaler/encoders (loaded via Joblib), MySQL connection for live aggregate queries.

## 25. Page-by-Page UI Design

- **Overview**: KPI cards (Average Burnout Risk, High-Risk Students %, Average Sleep, Average Academic Pressure) + top charts (risk distribution, risk by year).
- **Analytics**: filters (year, course, gender) + drill-down charts (burnout vs sleep, study hours, screen time, attendance; academic pressure & lifestyle balance distributions).
- **My Risk**: input form → predicted score, category, gauge/badge, top contributing factors, disclaimer.
- **What-If Simulator**: sliders for editable inputs, live-recalculated risk, current vs new comparison, delta.
- **Risk Factors**: global feature importance chart, correlation view.
- **Model Insights**: model comparison table, confusion matrices, metric charts, methodology notes and limitations.

## 26. SQL Database Architecture

MySQL relational schema separating identity, academic, lifestyle, and prediction data to demonstrate normalized design and JOIN-based analytics.

## 27. Database Schema

```
students(student_id PK, age, gender, year_of_study, course)
academic_data(student_id FK, attendance_percentage, cgpa, study_hours_per_day,
               assignment_workload, exam_frequency, academic_pressure)
lifestyle_data(student_id FK, sleep_hours, sleep_quality, screen_time_hours,
               physical_activity_hours, social_interaction_hours,
               breaks_per_day, hobbies_hours_per_week, days_off_per_week)
burnout_predictions(student_id FK, academic_pressure_score, lifestyle_balance_score,
               burnout_score, burnout_risk, model_version, predicted_at)
```

## 28. SQL Analytics Requirements

Queries must use SELECT, WHERE, GROUP BY, ORDER BY, JOIN, COUNT, AVG, CASE, and at least one subquery — e.g., % of students at high risk per course, average burnout by year, sleep-bucketed average risk via CASE, students above the average academic pressure via subquery.

## 29. What-If Simulator

User adjusts editable inputs (sleep, study hours, screen time, etc.) in the UI; the app re-runs the persisted model on the modified feature vector and displays Current Risk, New Risk, and Difference, with a disclaimer that this is a model-based simulation, not a guaranteed outcome.

## 30. Recommendation Logic

Rule-based, model-informed observations (not medical advice), e.g., comparing the student's inputs to dataset averages/medians and to the feature-importance ranking — framed strictly as data observations.

## 31. Error Handling

- Input validation on all form fields (range checks, required fields).
- Graceful handling of missing DB connection (fallback to cached CSV).
- Try/except around model loading and prediction with user-facing friendly messages.

## 32. Privacy and Ethics

Uses synthetic or public anonymized data only; no real student PII. `student_id` is a synthetic identifier. Explicit in-app disclaimer that results are for educational/demonstration purposes only and are not medical advice.

## 33. Responsible AI Considerations

Feature importance is presented as association, not causation. No claims of diagnostic accuracy. Model limitations (dataset size, synthetic nature, generalizability) documented in Model Insights.

## 34. MVP Scope

Cleaned dataset, MySQL integration, EDA, feature engineering, 2-model comparison with real metrics, prediction with explainability, Streamlit dashboard (all six pages), What-If Simulator, rule-based recommendations.

## 35. Optional Features

Downloadable PDF/CSV report, a third ML model, additional demographic breakdowns, dark/light theme toggle.

## 36. Future Improvements

Real longitudinal data collection (with consent), SHAP-based explainability, model retraining pipeline, deployment to a hosted Streamlit instance with authentication.

## 37. Deployment Plan

Local run via `streamlit run app.py` against a local MySQL instance for the MVP; optional Streamlit Community Cloud deployment with a cloud-hosted MySQL (e.g., PlanetScale/Railway) as a stretch goal.

## 38. Testing Strategy

Unit checks on feature engineering functions and scoring thresholds; manual QA of each dashboard page; validation that What-If deltas move in the expected direction for known input changes; SQL query result spot-checks against Pandas equivalents.

## 39. Success Criteria

All six dashboard pages functional; model metrics realistic (not near-perfect); What-If Simulator produces sensible directional changes; SQL analytics demonstrably used (not decorative); full pipeline traceable from raw data to dashboard; project completed as a working MVP within ~2 days.
