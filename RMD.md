# MindMap AI — RMD (Requirements + Module Design)

Companion document to `design.md`. Translates the product design into concrete, testable development requirements for MindMap AI (Student Burnout Analytics & Predictive Risk Assessment Platform).

---

## 1. Functional Requirements

- FR1: System shall load and clean the student dataset (raw CSV → cleaned dataset).
- FR2: System shall store cleaned data in a normalized MySQL schema (`students`, `academic_data`, `lifestyle_data`, `burnout_predictions`).
- FR3: System shall compute derived features: academic_pressure_score, lifestyle_balance_score.
- FR4: System shall train and compare at least two ML models (Logistic Regression, Random Forest).
- FR5: System shall persist the best/selected model via Joblib.
- FR6: System shall accept individual student inputs and return a burnout risk score (0–100) and category.
- FR7: System shall display top contributing factors per prediction.
- FR8: System shall provide a What-If Simulator that recalculates risk on modified inputs.
- FR9: System shall provide cohort-level analytics with filters (year, course, gender).
- FR10: System shall run SQL queries (via MySQL) to power aggregate analytics views.
- FR11: System shall generate non-medical, data-driven recommendations.
- FR12: System shall display model comparison metrics and confusion matrices.
- FR13 (optional): System shall allow downloading a summary report.

## 2. Non-Functional Requirements

- NFR1: Dashboard pages should load within a few seconds on local hardware with the target dataset size (thousands of rows, not millions).
- NFR2: Code should be modular — data, features, model, and UI logic separated into distinct modules/files.
- NFR3: No hard-coded predictions; all outputs must come from the trained model.
- NFR4: No fabricated or artificially inflated metrics; metrics must reflect actual evaluation.
- NFR5: UI must clearly and repeatedly disclose the non-medical, educational nature of predictions.
- NFR6: System must run without paid APIs, external LLMs, or authentication systems.

## 3. User Stories

- As a student, I want to enter my academic and lifestyle details so that I can see my predicted burnout risk.
- As a student, I want to see which factors most influenced my risk score so I understand what's driving it.
- As a student, I want to try changing my sleep/study/screen-time inputs so I can see how my risk might shift.
- As an advisor, I want to filter burnout analytics by year and course so I can spot at-risk cohorts.
- As a reviewer, I want to see model comparison metrics so I can judge the rigor of the ML work.

## 4. Acceptance Criteria

**FEATURE: Burnout Risk Prediction**
- Inputs are validated (ranges enforced, required fields checked).
- Missing/invalid values are handled gracefully (blocked submission or safe defaults, per field).
- Model generates a prediction from the trained pipeline (no hard-coded output).
- Risk probability is converted to a 0–100 score.
- Risk category (Low/Moderate/High/Very High) is displayed alongside the score.
- Top 3–5 contributing features are displayed with an "influence, not causation" disclaimer.
- A non-medical disclaimer is visible on the same screen as the prediction.

**FEATURE: What-If Simulator**
- User can adjust at least sleep_hours, study_hours_per_day, and screen_time_hours.
- Current risk and new risk are both displayed, with the numeric difference.
- Recalculation uses the same persisted model/pipeline as the main prediction (no separate ad hoc logic).
- A "model-based simulation, not a guaranteed outcome" disclaimer is shown.

**FEATURE: Cohort Analytics**
- Filters for year_of_study, course, and gender are available and functional.
- Charts update reactively to filter selections.
- At least: risk distribution, risk by year, burnout vs sleep, burnout vs study hours, burnout vs screen time charts are present.

**FEATURE: SQL Analytics**
- At least one query each using GROUP BY + AVG, JOIN across two tables, CASE-based bucketing, and a subquery.
- Query results match equivalent Pandas computations (spot-checked in testing).

**FEATURE: Model Comparison**
- Logistic Regression and Random Forest are both trained on the same train/test split.
- Accuracy, Precision, Recall, F1, and Confusion Matrix are computed and displayed for each.
- No metric is manually overridden or fabricated.

## 5. Data Requirements

- Dataset must include all fields listed in design.md §14 (Data Dictionary).
- Relationships (e.g., lower sleep → generally higher risk) must be present but non-deterministic (added noise).
- Minimum viable size: a few thousand synthetic student records for stable EDA/ML results.

## 6. Database Requirements

- MySQL instance with the four-table schema in design.md §27.
- Foreign keys from `academic_data`, `lifestyle_data`, `burnout_predictions` to `students.student_id`.
- Data loaded via a repeatable load script (idempotent — safe to re-run).

## 7. SQL Requirements

- Demonstrated use of SELECT, WHERE, GROUP BY, ORDER BY, JOIN, COUNT, AVG, CASE, and at least one subquery, as functioning analytics (not decorative).

## 8. Analytics Requirements

- Must answer, at minimum: % of students at high/very-high risk; average risk by year and by course; sleep vs burnout relationship; study hours vs burnout relationship; screen time vs burnout relationship; attendance vs burnout relationship; top factors associated with high risk.

## 9. ML Requirements

- Preprocessing pipeline (imputation, encoding, scaling) must be reusable at both training and inference time (same transformer objects, persisted).
- Train/test split must be stratified on `burnout_risk`.
- At least two models trained and compared; best model (or a clearly justified choice) persisted for the app.

## 10. UI Requirements

- Streamlit multipage app with the six pages defined in design.md §24.
- Consistent visual language (premium, minimal, modern) per design.md §11.
- All charts built with Plotly for interactivity.

## 11. Prediction Requirements

- Prediction must be generated live from user input through the persisted pipeline — never precomputed/hard-coded per input combination.

## 12. Explainability Requirements

- Every individual prediction must show top contributing factors.
- A global feature-importance view must exist on the "Risk Factors" page.
- All explainability language must avoid causal claims.

## 13. What-If Simulator Requirements

- See Acceptance Criteria above (§4). Must reuse the production model/pipeline.

## 14. Recommendation Requirements

- Recommendations must be derived from comparing user inputs to dataset statistics and/or feature importance — never generic filler unconnected to the data.
- Must avoid any medical/clinical language.

## 15. Error/Validation Requirements

- All numeric inputs bounded to realistic ranges (per Data Dictionary) with inline validation messages.
- Categorical inputs restricted to known valid values.
- Database connection failures must not crash the app — fallback to cached/local data with a visible notice.

## 16. Security/Privacy Requirements

- No real student PII collected or stored.
- Synthetic `student_id` values only.
- No authentication/user-account system required for MVP (out of scope per design.md §6).

## 17. Testing Requirements

- Unit tests for feature engineering functions (academic_pressure_score, lifestyle_balance_score calculations).
- Unit tests for risk-category thresholding (0–30/31–60/61–80/81–100).
- Manual QA checklist for each of the six dashboard pages.
- Sanity check that What-If Simulator moves risk in the expected direction for at least 3 known input changes (e.g., increasing sleep should not increase risk, all else equal, in the large majority of test cases).

## 18. Performance Requirements

- Cohort analytics views should render without noticeable lag for the target dataset size using cached data loading (e.g., Streamlit `@st.cache_data`).
- Model inference for a single prediction should return in well under a second.

## 19. Module Breakdown

1. `data/` — raw and cleaned dataset artifacts, data generation/cleaning scripts.
2. `db/` — MySQL schema DDL, data-loading scripts, query functions.
3. `features/` — feature engineering functions (academic_pressure_score, lifestyle_balance_score).
4. `ml/` — preprocessing pipeline, model training, evaluation, persistence (Joblib), prediction/explainability functions.
5. `app/` — Streamlit multipage application (Overview, Analytics, My Risk, What-If Simulator, Risk Factors, Model Insights).
6. `utils/` — shared validation, formatting, and disclaimer-text helpers.

## 20. Dependencies Between Modules

`data/` → `db/` (loading) → `features/` (reads cleaned/DB data) → `ml/` (consumes engineered features) → `app/` (consumes both `db/` for analytics and `ml/` for prediction/simulation). `utils/` is used across `app/`, `ml/`, and `features/`.

## 21. MVP vs Optional Requirements

**MVP:** FR1–FR12, all Acceptance Criteria in §4, six-page dashboard, two-model comparison, SQL analytics, What-If Simulator, explainability, recommendations.

**Optional:** FR13 (downloadable report), a third ML model, additional demographic breakdowns, theme toggle, cloud deployment.

## 22. Definition of Done

- All MVP functional requirements implemented and manually verified against their acceptance criteria.
- Dashboard runs end-to-end without errors from a fresh environment (`streamlit run app.py`).
- Model metrics are real, documented, and within a believable (non-perfect) range.
- SQL queries verified against Pandas equivalents.
- Disclaimers present on all prediction/simulation/recommendation surfaces.
- design.md, RMD.md, and initial_prompt.md remain internally consistent with the shipped implementation.
