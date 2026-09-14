-- =====================================================================
-- MindMap AI — MySQL Relational Database Schema DDL
-- Normalized design separating demographics, academics, lifestyle, and predictions
-- per design.md §26-27 and RMD.md §2
-- =====================================================================

CREATE DATABASE IF NOT EXISTS mindmap_ai;
USE mindmap_ai;

-- 1. Students Table (Demographics & Cohort Context)
CREATE TABLE IF NOT EXISTS students (
    student_id VARCHAR(16) PRIMARY KEY,
    age INT NOT NULL,
    gender VARCHAR(16) NOT NULL,
    year_of_study INT NOT NULL,
    course VARCHAR(64) NOT NULL
);

-- 2. Academic Data Table (Workload & Performance)
CREATE TABLE IF NOT EXISTS academic_data (
    student_id VARCHAR(16) PRIMARY KEY,
    attendance_percentage DECIMAL(5, 2) NOT NULL,
    cgpa DECIMAL(4, 2) NOT NULL,
    study_hours_per_day DECIMAL(4, 2) NOT NULL,
    assignment_workload INT NOT NULL,
    exam_frequency INT NOT NULL,
    academic_pressure INT NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
);

-- 3. Lifestyle Data Table (Sleep, Screen Time, Activity & Wellness)
CREATE TABLE IF NOT EXISTS lifestyle_data (
    student_id VARCHAR(16) PRIMARY KEY,
    sleep_hours DECIMAL(4, 2) NOT NULL,
    sleep_quality INT NOT NULL,
    screen_time_hours DECIMAL(4, 2) NOT NULL,
    physical_activity_hours DECIMAL(4, 2) NOT NULL,
    social_interaction_hours DECIMAL(4, 2) NOT NULL,
    breaks_per_day INT NOT NULL,
    hobbies_hours_per_week DECIMAL(4, 2) NOT NULL,
    days_off_per_week INT NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
);

-- 4. Burnout Predictions Table (Scores, Targets & Metadata)
CREATE TABLE IF NOT EXISTS burnout_predictions (
    student_id VARCHAR(16) PRIMARY KEY,
    academic_pressure_score DECIMAL(5, 2) NOT NULL,
    lifestyle_balance_score DECIMAL(5, 2) NOT NULL,
    burnout_score DECIMAL(5, 2) NOT NULL,
    burnout_risk VARCHAR(16) NOT NULL,
    model_version VARCHAR(32) DEFAULT 'v1.0.0',
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
);

-- =====================================================================
-- Multi-User Student & Admin Application Tables
-- =====================================================================

-- 5. Users Table (Authentication & Role Management)
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(128) NOT NULL,
    email VARCHAR(128) UNIQUE NOT NULL,
    password_hash VARCHAR(128) NOT NULL,
    salt VARCHAR(64) NOT NULL,
    role VARCHAR(16) NOT NULL DEFAULT 'student',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. Student Profiles Table (Personal Student Demographics & Academic Context)
CREATE TABLE IF NOT EXISTS student_profiles (
    profile_id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) UNIQUE NOT NULL,
    age INT NOT NULL DEFAULT 21,
    gender VARCHAR(16) NOT NULL DEFAULT 'Other',
    year_of_study INT NOT NULL DEFAULT 2,
    course VARCHAR(64) NOT NULL DEFAULT 'Computer Science',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- 7. Assessments Table (Submitted Multi-Section Student Assessments)
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

-- 8. Predictions Table (Prediction Outcomes & Historical Analytics)
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


