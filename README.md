# 🧠 MindMap AI — Student Burnout Analytics & Predictive Risk Assessment Platform

> *"Understand the Pattern. Predict the Risk. Take Back Control."*

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Database](https://img.shields.io/badge/Database-MySQL%20%2F%20SQLite-teal.svg)](https://www.mysql.com/)
[![Tests](https://img.shields.io/badge/Tests-36%20Passed%20(100%25)-brightgreen.svg)](https://pytest.org/)

---

## 📌 1. Project Overview

**MindMap AI** is an end-to-end data science product designed to analyze academic workloads, physiological recovery indicators, and lifestyle patterns among students to surface early burnout risk signals. 

Rather than functioning as a simplistic, toy machine learning exercise or a black-box model, MindMap AI provides transparent, explainable predictive analytics backed by a **normalized 4-table SQL relational database**, automated feature engineering, multi-algorithm evaluation, and an interactive **Streamlit** dashboard.

---

## 🎯 2. Key Features

1. **📊 Cohort Overview & KPI Dashboard:**
   - Real-time aggregate KPIs (Average Burnout Score, High-Risk Cohort %, Average Sleep Duration, Academic Pressure Index, Lifestyle Balance Index).
   - Interactive Donut charts and academic year progression breakdowns.
   - SQL-backed course cohort comparison table.

2. **📈 Multi-Dimensional Cohort Analytics:**
   - Dynamic sidebar filters (Course, Academic Year, Gender) with instant cross-chart reactivity.
   - Scatter plots with Ordinary Least Squares (OLS) linear trendlines (Sleep vs. Burnout, Screen Time vs. Burnout, Attendance vs. CGPA).
   - 2D Quadrant Analysis mapping students into *Thriving*, *Strained*, and *Severe Vulnerability* segments.
   - SQL `CASE WHEN` sleep duration tier aggregations and live SQL subquery inspections.

3. **🎯 Individual Burnout Risk Assessment (My Risk):**
   - Form for entering personalized academic and lifestyle parameters.
   - Live model inference outputting a continuous Burnout Risk Score ($0–100$), Radial Gauge visualization, and Risk Category badge (`Low`, `Moderate`, `High`, `Very High`).
   - Multi-class probability distribution ($P(\text{Low}), P(\text{Moderate}), P(\text{High}), P(\text{Very High})$).
   - Dynamic factor attribution cards categorizing **Stress Drivers (🔺)** vs. **Protective Buffers (🛡️)** benchmarked against cohort medians.
   - Data-driven lifestyle and study pacing recommendations.

4. **🔮 Interactive What-If Simulator:**
   - Side-by-side dual habit controllers (Sleep, Sleep Quality, Study Hours, Screen Time, Exercise, Breaks, Hobbies, Days Off).
   - Real-time risk delta ($\Delta$) scoring and grouped comparative bar chart.
   - Quick presets (*Stressed Tech Student*, *Overworked Pre-Med Student*, *Balanced Freshman*).

5. **🧬 Global Risk Factors & Explainability:**
   - Global feature importance rankings comparing Random Forest, Logistic Regression, and Gradient Boosting.
   - Full $18 \times 18$ annotated Pearson correlation matrix heatmap.
   - Non-causal responsible AI explainability framework.

6. **🔬 ML Model Insights & Methodology:**
   - Transparent performance metrics (Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1).
   - Multi-class confusion matrices and per-class classification reports.
   - Full methodology documentation and ethical considerations.

---

## 🏗️ 3. System Architecture & Relational Schema

```
Student Dataset (3,000 Records)
     │
     ├──> Data Cleaning & Audit (data/clean_data.py)
     │         │
     │         ├──> Normalized Relational Database (MySQL / SQLite Fallback: 4 Tables)
     │         │         ├── students (student_id PK, age, gender, year_of_study, course)
     │         │         ├── academic_data (student_id FK, attendance, cgpa, study_hours, workload, exam_freq, pressure)
     │         │         ├── lifestyle_data (student_id FK, sleep_hours, sleep_quality, screen_time, exercise, breaks, hobbies)
     │         │         └── burnout_predictions (student_id FK, pressure_score, balance_score, burnout_score, burnout_risk)
     │         │
     │         └──> Feature Engineering (features/engineering.py)
     │                   │
     │                   ├──> ML Training & Comparison (ml/train.py)
     │                   │         │
     │                   │         └──> Persisted Joblib Pipeline (ml/models/best_model.joblib)
     │                   │                   │
     │                   │                   └──> Live Inference & What-If Engine (ml/predict.py)
     │                   │
     │                   └──> SQL Analytics Engine (db/queries.py)
     │
     └──> Streamlit Multipage Dashboard (app.py & pages/)
```

---

## 🔬 4. Machine Learning Evaluation & Comparison

Evaluated on a held-out test set ($N = 600$ stratified student samples):

| Model | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall | Primary Driving Features |
|---|---|---|---|---|---|---|
| **Logistic Regression** *(Selected Best)* | **$77.00\%$** | **$0.6307$** | **$0.7656$** | $0.6774$ | $0.6062$ | `assignment_workload`, `academic_pressure_score`, `academic_pressure`, `sleep_quality` |
| **Random Forest** | $77.50\%$ | $0.5440$ | $0.7639$ | $0.5635$ | $0.5364$ | `academic_pressure_score`, `lifestyle_balance_score`, `academic_pressure`, `sleep_quality` |
| **Gradient Boosting** | $75.17\%$ | $0.5336$ | $0.7486$ | $0.5431$ | $0.5302$ | `academic_pressure_score`, `lifestyle_balance_score`, `attendance_percentage` |

> **Note on Realistic Model Accuracy:** Models achieve realistic, credible accuracy ($\approx 75–78\%$) reflecting real-world behavioral variance and noise rather than an artificially inflated $100\%$ score.

---

## 🚀 5. Quick Start & Run Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- MySQL Server (optional; automated SQLite fallback ensures 100% zero-configuration local execution)

### 1. Clone & Setup Environment
```bash
# Clone repository
git clone https://github.com/your-username/mindmap-ai.git
cd mindmap-ai

# Create & activate virtual environment (optional)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (Optional for MySQL)
Copy the environment template:
```bash
cp .env.example .env
```
*(If running against local MySQL, specify your `DB_USER` and `DB_PASSWORD` in `.env`. If left unconfigured, the app automatically runs on the included embedded database seamlessly).*

### 3. Generate Data, Clean & Train Models
```bash
# 1. Generate synthetic raw data (3,000 records)
python data/generate_dataset.py

# 2. Clean data & run exploratory data analysis
python data/clean_data.py
python data/eda.py

# 3. Load data into relational database
python db/load_data.py

# 4. Train and evaluate ML models
python ml/train.py
```

### 4. Launch the Interactive Dashboard
```bash
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 🧪 6. Automated Testing Suite

MindMap AI includes a test suite with **36 automated unit tests** across 5 test suites:

```bash
pytest -v
```

### Test Coverage Breakdown:
- `tests/test_features.py` (22 tests): Verifies mathematical bounds, strict monotonicity, and risk thresholding.
- `tests/test_prediction.py` (3 tests): Tests live pipeline inference, probability summation, and explainability attributions.
- `tests/test_recommendations.py` (3 tests): Validates trigger rules for sleep, digital wellness, and resilience maintenance.
- `tests/test_simulator.py` (4 tests): Tests directional sensitivity across What-If lifestyle adjustments.
- `tests/test_sql_validation.py` (4 tests): Spot-checks SQL query outputs directly against Pandas calculations.

---

## 📂 7. Project Structure

```
mindmap-ai/
├── app.py                         # Master Streamlit entry point
├── config.py                      # Centralized configuration, paths & constants
├── requirements.txt               # Pinned Python dependencies
├── .env.example                   # Environment variable template
├── .gitignore                     # Git exclusion rules
├── design.md                      # Master Product & Technical Design Document
├── RMD.md                         # Requirements & Module Design Document
├── initial_prompt.md              # Implementation Plan & Phase Milestones
│
├── pages/                         # Streamlit Multipage Views
│   ├── 1_📊_Overview.py           # Cohort KPIs, distributions & year trends
│   ├── 2_📈_Analytics.py          # Filterable drill-downs & SQL CASE bucketing
│   ├── 3_🎯_My_Risk.py            # Live prediction form & factor attributions
│   ├── 4_🔮_What_If_Simulator.py  # Interactive lifestyle simulation sliders
│   ├── 5_🧬_Risk_Factors.py       # Global feature importance & correlation matrix
│   └── 6_🔬_Model_Insights.py     # Multi-model evaluation & confusion matrices
│
├── data/                          # Data Management
│   ├── generate_dataset.py        # Realistic synthetic dataset generator
│   ├── clean_data.py              # Validation, de-duplication & imputation
│   ├── eda.py                     # Univariate, bivariate & ANOVA statistical analysis
│   ├── raw/                       # Raw CSV storage
│   └── processed/                 # Cleaned CSV & summary JSON artifacts
│
├── db/                            # Relational Database Integration
│   ├── schema.sql                 # MySQL 4-table relational DDL
│   ├── connection.py              # MySQL & SQLite connection manager
│   ├── load_data.py               # Idempotent database loader
│   └── queries.py                 # SQL analytical query suite
│
├── features/                      # Feature Engineering
│   ├── engineering.py             # Academic pressure & lifestyle balance formulas
│   ├── definitions.py             # Schema metadata, bounds & labels
│   └── pipeline.py                # Enriched feature transformer
│
├── ml/                            # Machine Learning Pipeline
│   ├── train.py                   # Preprocessing, multi-model training & evaluation
│   ├── predict.py                 # Live inference & explainability engine
│   └── models/                    # Persisted Joblib models & metrics JSON
│
├── utils/                         # Shared Utilities
│   ├── ui_components.py           # CSS styling, headers, KPI cards & badges
│   └── recommendations.py         # Data-driven non-medical recommendation engine
│
└── tests/                         # Automated Unit Tests
    ├── test_features.py           # Feature engineering tests
    ├── test_prediction.py         # Prediction pipeline tests
    ├── test_recommendations.py    # Recommendation rule tests
    ├── test_simulator.py          # What-If sensitivity tests
    └── test_sql_validation.py     # SQL vs. Pandas spot-check tests
```

---

## ⚖️ 8. Responsible AI & Ethical Framework

- **Educational & Portfolio Scope:** MindMap AI is strictly an analytics and machine learning demonstration platform. It does **not** provide clinical diagnosis, psychiatric assessment, or medical treatment recommendations.
- **Statistical Association vs. Causation:** Feature importance metrics and risk factors denote observed statistical patterns across the student dataset, avoiding definitive causal assertions.
- **Zero Real PII:** All records use synthetic non-identifiable identifiers (`student_id: STU0001` to `STU3000`).

---

## 📄 9. License

Distributed under the MIT License. See `LICENSE` for more information.
