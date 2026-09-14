# MindMap AI — Master Implementation Prompt

This is the prompt to give to an AI coding agent (inside Antigravity or similar) to actually build MindMap AI: Student Burnout Analytics & Predictive Risk Assessment Platform.

---

## PROJECT CONTEXT

You are building **MindMap AI**, a Student Burnout Analytics & Predictive Risk Assessment Platform. This is a portfolio-grade data science product, not a toy notebook project. Two governing documents already exist in this repository:

- `design.md` — the master product and technical design document
- `RMD.md` — the requirements and module design document

**Read `design.md` and `RMD.md` in full before writing any code.** Everything you build must be consistent with them. If something is ambiguous, prefer the simpler, more reliable interpretation over an elaborate one.

## TECH STACK

Python, Pandas, NumPy, Scikit-learn, MySQL, SQL, Plotly, Streamlit, Joblib. Matplotlib/Seaborn only for exploratory/dev-time plots if useful. Do NOT introduce deep learning, LLM APIs, paid APIs, chatbots, authentication systems, microservices, Kubernetes, or unnecessary cloud infrastructure.

## CODING PRINCIPLES

- Do not contradict `design.md`.
- Do not invent unnecessary features beyond `design.md` / `RMD.md`.
- Do not overengineer — this is a ~2-day MVP.
- Build incrementally, phase by phase (see below).
- Keep the application runnable after each major phase.
- Prefer simple, reliable solutions over clever ones.
- Never hard-code predictions — all outputs must come from the trained model pipeline.
- Never fake or artificially inflate ML metrics.
- Never claim medical diagnosis anywhere in code, UI copy, or comments — burnout is a "risk score," not a diagnosis.
- Use realistic, non-deterministic synthetic data (see Dataset Requirements).
- Keep code modular and readable, following the module breakdown in `RMD.md` §19.
- Explain major implementation decisions in code comments or a brief note after each phase.
- **Stop and report progress after each phase below rather than generating all code at once.**

## DATASET REQUIREMENTS

- Generate a synthetic dataset matching the schema in `design.md` §14 (Data Dictionary).
- Build in realistic-but-noisy relationships (e.g., lower sleep tends toward higher burnout, higher academic pressure tends toward higher burnout) using randomness/noise so relationships are not perfectly deterministic.
- Target a resulting model accuracy in a believable range — flag it as a problem if any model scores at or near 99–100% accuracy, since that indicates a leaky/deterministic target.
- Save the cleaned dataset in a form that both the ML pipeline and the MySQL loader can consume.

## SQL REQUIREMENTS

- Implement the schema in `design.md` §27 in MySQL.
- Write a loader script that populates `students`, `academic_data`, `lifestyle_data`, `burnout_predictions` from the cleaned dataset, safely re-runnable.
- Implement the analytics queries required in `RMD.md` §7–8, using SELECT/WHERE/GROUP BY/ORDER BY/JOIN/COUNT/AVG/CASE, and at least one subquery.
- Wire these queries into the Streamlit Analytics/Overview pages as the actual data source for cohort-level aggregates (not decorative).

## ML REQUIREMENTS

- Build a preprocessing pipeline (imputation, encoding, scaling) as reusable, persisted transformer objects — the same pipeline must be used for training and for live prediction/What-If simulation.
- Train and compare Logistic Regression and Random Forest (optionally a third simple model).
- Evaluate with Accuracy, Precision, Recall, F1, and Confusion Matrix; persist metrics for display in Model Insights.
- Persist the selected model (and preprocessing pipeline) via Joblib.
- Implement explainability: feature importances (Random Forest) and/or coefficients (Logistic Regression), plus a per-prediction "top contributing factors" extraction.

## UI REQUIREMENTS

- Build a Streamlit multipage app with these pages: Overview, Analytics, My Risk, What-If Simulator, Risk Factors, Model Insights (per `design.md` §24–25).
- Visual style: premium, minimal, modern, data-focused — not childish or over-animated.
- Use Plotly for all interactive charts.

## DASHBOARD REQUIREMENTS

- Overview: KPI cards (Average Burnout Risk, High-Risk Students %, Average Sleep, Average Academic Pressure) plus top summary charts.
- Analytics: filters by year_of_study/course/gender, with drill-down relationship charts.
- Risk Factors: global feature importance and correlation views.
- Model Insights: model comparison table, confusion matrices, metric charts, and a short methodology/limitations note.

## WHAT-IF SIMULATOR

- Allow editing at least sleep_hours, study_hours_per_day, and screen_time_hours via sliders.
- Recompute risk through the same persisted pipeline used for the main prediction.
- Display Current Risk, New Risk, and the numeric Difference.
- Include a disclaimer: "This is a model-based simulation, not a guaranteed real-world outcome."

## EXPLAINABILITY

- On the "My Risk" page, show the top 3–5 factors contributing to that individual's prediction, phrased as influence rather than causation.
- On the "Risk Factors" page, show global feature importance across the trained model.

## ERROR HANDLING

- Validate all user inputs against realistic ranges from the Data Dictionary; show inline errors for invalid input.
- Handle MySQL connection failures gracefully with a fallback to a cached local dataset and a visible notice to the user.
- Wrap model loading and prediction calls in try/except with clear, non-technical user-facing error messages.

## TESTING

- Add unit tests for the feature engineering functions (academic_pressure_score, lifestyle_balance_score) and for risk-category thresholding.
- Manually verify each dashboard page against the acceptance criteria in `RMD.md` §4.
- Verify at least 3 What-If scenarios move risk in the expected direction.
- Spot-check SQL query outputs against equivalent Pandas computations.

## DOCUMENTATION

- Add a `README.md` only once implementation begins (not in this initial phase) explaining setup, run instructions, and project structure.
- Keep inline comments explaining non-obvious feature engineering and modeling decisions.

## RUN INSTRUCTIONS (target end-state)

- `streamlit run app.py` should launch the full dashboard locally against a configured local MySQL instance (connection details via environment variables/config, not hard-coded credentials).

---

## IMPLEMENTATION ORDER (PHASES)

**PHASE 1 — Project structure and environment**
Set up folder structure per `RMD.md` §19 (`data/`, `db/`, `features/`, `ml/`, `app/`, `utils/`), environment/dependency setup. Report back before continuing.

**PHASE 2 — Dataset creation/loading**
Generate the synthetic dataset per the Data Dictionary and dataset requirements above. Report back.

**PHASE 3 — Data cleaning and EDA**
Clean the dataset per `design.md` §15, run and save/summarize EDA per §16–17. Report back.

**PHASE 4 — MySQL database**
Create schema, write and run the loader script. Report back.

**PHASE 5 — Feature engineering**
Implement academic_pressure_score and lifestyle_balance_score. Report back.

**PHASE 6 — ML training and evaluation**
Build the preprocessing pipeline, train Logistic Regression and Random Forest, evaluate and compare. Report back with real metrics.

**PHASE 7 — Prediction pipeline**
Wire up single-input prediction with explainability using the persisted model. Report back.

**PHASE 8 — Streamlit dashboard**
Scaffold the six-page multipage app shell. Report back.

**PHASE 9 — Analytics visualizations**
Implement Overview and Analytics pages using SQL-backed queries and Plotly charts. Report back.

**PHASE 10 — Explainability**
Implement the Risk Factors page and per-prediction contributing factors on My Risk. Report back.

**PHASE 11 — What-If Simulator**
Implement the simulator page per requirements above. Report back.

**PHASE 12 — Recommendations**
Implement rule-based, data-driven recommendation text on My Risk. Report back.

**PHASE 13 — Testing and polishing**
Add unit tests, run manual QA against `RMD.md` §4 acceptance criteria, polish UI/UX. Report back.

**PHASE 14 — Final documentation**
Write `README.md`, run instructions, and a summary of any deviations from `design.md`/`RMD.md` (with justification). Report back.

**Stop and report progress after each phase. Do not proceed to the next phase without confirming the current one is complete and runnable.**
