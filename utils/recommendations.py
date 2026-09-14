"""
MindMap AI - Data-Driven Recommendations Engine
Generates model-informed, non-medical behavioral observations by benchmarking
individual student metrics against cohort statistics per design.md §30 and RMD.md §14.
"""

from typing import Dict, Any, List


def generate_data_driven_recommendations(
    student_data: Dict[str, Any],
    prediction_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Generates structured, data-driven lifestyle and study pacing recommendations
    based on the student's deviations from cohort benchmarks and feature importance rankings.

    Parameters:
    - student_data: raw input dictionary of student parameters
    - prediction_result: output dictionary from predict_burnout_risk()

    Returns:
    - List of recommendation dictionaries with priority, category, title, text, and benchmark data.
    """
    recommendations = []

    sleep_h = float(student_data.get("sleep_hours", 7.0))
    sleep_q = int(student_data.get("sleep_quality", 7))
    screen_h = float(student_data.get("screen_time_hours", 7.0))
    study_h = float(student_data.get("study_hours_per_day", 5.0))
    breaks = int(student_data.get("breaks_per_day", 4))
    phys_act = float(student_data.get("physical_activity_hours", 4.0))
    days_off = int(student_data.get("days_off_per_week", 1))
    workload = int(student_data.get("assignment_workload", 5))

    risk_score = float(prediction_result.get("risk_score", 50.0))
    pressure_score = float(prediction_result.get("academic_pressure_score", 50.0))
    balance_score = float(prediction_result.get("lifestyle_balance_score", 50.0))

    # 1. Sleep & Recovery Recommendation
    if sleep_h < 6.0 or sleep_q < 5:
        recommendations.append({
            "category": "Sleep & Recovery",
            "priority": "High Impact",
            "icon": "🛌",
            "title": "Restore Nightly Sleep Duration & Sleep Quality",
            "observation": (
                f"You currently average {sleep_h:.1f} hours of sleep (quality: {sleep_q}/10). "
                "Cohort SQL analysis indicates students sleeping < 5.5 hours exhibit a 66.3% high-risk rate, "
                "whereas students averaging 7.0–8.5 hours show only a 2.9% high-risk rate. "
                "Aiming for 7.0+ hours is the single highest-impact protective lifestyle buffer in the model."
            ),
            "suggested_action": "Establish a consistent 30-minute wind-down routine and aim for 7.0–8.0 hours of nightly rest.",
        })

    # 2. Study Pacing & Structured Breaks
    if breaks < 3 and study_h >= 5.0:
        recommendations.append({
            "category": "Study Pacing",
            "priority": "High Impact" if study_h >= 7.5 else "Moderate Impact",
            "icon": "⏱️",
            "title": "Introduce Structured Study Rest Intervals",
            "observation": (
                f"You report {study_h:.1f} daily study hours with only {breaks} breaks per day. "
                "In the dataset, study breaks have a strong negative correlation with burnout (r = -0.67). "
                "Continuous cognitive exertion without rest blocks significantly increases self-reported strain."
            ),
            "suggested_action": "Apply the Pomodoro or 50/10 technique: take a 5-10 minute screen-free break every 50 minutes of focused study.",
        })

    # 3. Screen Time Management
    if screen_h >= 8.5:
        recommendations.append({
            "category": "Digital Wellness",
            "priority": "Moderate Impact",
            "icon": "📱",
            "title": "Reduce Non-Essential Screen Exposure",
            "observation": (
                f"Your daily screen time of {screen_h:.1f} hours places you in the top quartile of student screen exposure. "
                "High screen time correlates positively with burnout (r = +0.58) and negatively with sleep quality (r = -0.45)."
            ),
            "suggested_action": "Curfew recreational screen usage 45 minutes prior to sleep to improve natural melatonin and sleep depth.",
        })

    # 4. Physical Activity & Stress Buffering
    if phys_act < 2.5:
        recommendations.append({
            "category": "Physical Health",
            "priority": "Moderate Impact",
            "icon": "🏃",
            "title": "Incorporate Regular Cardiovascular Exercise",
            "observation": (
                f"Your current physical activity is {phys_act:.1f} hrs/week (cohort average: 4.7 hrs/wk). "
                "Exercise ranks among the top 5 protective factors (r = -0.63) for modulating physiological cortisol levels."
            ),
            "suggested_action": "Incorporate 30–40 minutes of moderate activity (walking, jogging, cycling, or gym) 3–4 days per week.",
        })

    # 5. Dedicated Recovery Days
    if days_off == 0:
        recommendations.append({
            "category": "Work-Life Boundary",
            "priority": "High Impact",
            "icon": "🏖️",
            "title": "Designate at Least One Complete Weekly Recovery Day",
            "observation": (
                "Studying 7 days a week with zero full recovery days strongly accelerates compounding fatigue. "
                "Students taking 1–2 days off maintain comparable CGPAs while sustaining significantly lower burnout scores."
            ),
            "suggested_action": "Designate one weekend day or two half-days strictly dedicated to recovery, social connection, and personal hobbies.",
        })

    # 6. High Academic Workload De-escalation
    if workload >= 8 or pressure_score >= 65.0:
        recommendations.append({
            "category": "Academic Strategy",
            "priority": "High Impact",
            "icon": "📚",
            "title": "Chunk High-Volume Academic Assignments",
            "observation": (
                f"Your assignment workload ({workload}/10) and academic strain index ({pressure_score:.1f}/100) are severely elevated. "
                "Assignment workload is the #1 feature driving multi-class model classification."
            ),
            "suggested_action": "Break major deliverables into small daily milestones and engage study groups or academic advisory for workload pacing.",
        })

    # 7. Positive Reinforcement for Resilient Students
    if not recommendations or risk_score <= 30.0:
        recommendations.append({
            "category": "Maintenance",
            "priority": "Positive Reinforcement",
            "icon": "🌟",
            "title": "Maintain Balanced Lifestyle Architecture",
            "observation": (
                f"Your overall profile reflects strong lifestyle resilience (Balance Index: {balance_score:.1f}/100) "
                f"and controlled academic pressure ({pressure_score:.1f}/100), keeping your estimated burnout risk low ({risk_score:.1f}/100)."
            ),
            "suggested_action": "Continue your current balance of adequate sleep, regular physical activity, and study pacing throughout exam cycles.",
        })

    return recommendations
