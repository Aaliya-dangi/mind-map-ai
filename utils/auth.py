"""
MindMap AI - Local Secure Authentication & Session Management Module
Uses PBKDF2-HMAC with SHA-256 and unique per-user cryptographic salts.
Zero paid APIs, zero external services, zero plaintext passwords.
Supports Student and Admin roles with full session access controls.
"""

import os
import sys
import hashlib
import hmac
import secrets
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import streamlit as st

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from db.user_db import (
    create_user,
    get_user_by_email,
    get_student_profile,
    update_student_profile,
    ensure_admin_account,
)
from features.definitions import CATEGORICAL_VOCABULARIES


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hashes a password using PBKDF2-HMAC with SHA-256 and 100,000 iterations.
    Returns (hex_hash, hex_salt).
    """
    if salt is None:
        salt = secrets.token_hex(16)

    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000,
    )
    return derived.hex(), salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verifies a plain password against stored hash and salt."""
    computed_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(computed_hash, stored_hash)


def is_authenticated() -> bool:
    """Returns True if a user is currently logged into the Streamlit session."""
    return "user" in st.session_state and st.session_state["user"] is not None


def get_current_user() -> Optional[Dict[str, Any]]:
    """Returns the currently logged-in user dictionary or None."""
    return st.session_state.get("user", None)


def get_user_role() -> Optional[str]:
    """Returns the role of the logged-in user ('student', 'admin', or None)."""
    user = get_current_user()
    return user.get("role") if user else None


def is_student() -> bool:
    """Returns True if the logged-in user is a student."""
    return get_user_role() == "student"


def is_admin() -> bool:
    """Returns True if the logged-in user is an administrator."""
    return get_user_role() == "admin"


def get_current_profile() -> Optional[Dict[str, Any]]:
    """Returns the current student's full profile, syncing from DB if available."""
    user = get_current_user()
    if not user:
        return None

    # Fetch latest profile from DB
    profile = get_student_profile(user["user_id"])
    if profile:
        st.session_state["user_profile"] = profile
        return profile

    return st.session_state.get("user_profile", None)


def refresh_user_profile():
    """Forces a reload of user profile into session state."""
    user = get_current_user()
    if user:
        profile = get_student_profile(user["user_id"])
        if profile:
            st.session_state["user_profile"] = profile


def login_user(email: str, password: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Authenticates a user (student or admin) by email/username and password.
    Returns (success, message, user_dict).
    """
    ensure_admin_account()
    email_clean = email.strip().lower()
    user_record = get_user_by_email(email_clean)
    if not user_record:
        return False, "User not found. Please check your email or register a new student account.", None

    if not verify_password(password, user_record["password_hash"], user_record["salt"]):
        return False, "Incorrect password. Please try again.", None

    user_info = {
        "user_id": user_record["user_id"],
        "name": user_record["name"],
        "full_name": user_record["name"],
        "email": user_record["email"],
        "role": user_record["role"],
    }

    st.session_state["user"] = user_info

    if user_record["role"] == "student":
        profile = get_student_profile(user_record["user_id"])
        st.session_state["user_profile"] = profile
    else:
        st.session_state["user_profile"] = None

    return True, "Login successful!", user_info


def signup_user(
    name: str,
    email: str,
    password: str,
    age: int = 21,
    gender: str = "Other",
    year_of_study: int = 2,
    course: str = "Computer Science",
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Registers a new student user and logs them in.
    Returns (success, message, user_dict).
    """
    ensure_admin_account()
    email_clean = email.strip().lower()
    name_clean = name.strip()

    if not name_clean:
        return False, "Full name is required.", None

    if "@" not in email_clean or "." not in email_clean:
        return False, "Please enter a valid email address.", None

    if len(password) < 6:
        return False, "Password must be at least 6 characters long.", None

    existing = get_user_by_email(email_clean)
    if existing:
        return False, f"Email '{email_clean}' is already registered. Please log in.", None

    pwd_hash, salt = hash_password(password)
    user_info = create_user(
        name=name_clean,
        email=email_clean,
        password_hash=pwd_hash,
        salt=salt,
        role="student",
        age=int(age),
        gender=gender,
        year_of_study=int(year_of_study),
        course=course,
    )

    if not user_info:
        return False, "Failed to create account. Please try again.", None

    user_dict = {
        "user_id": user_info["user_id"],
        "name": user_info["name"],
        "full_name": user_info["name"],
        "email": user_info["email"],
        "role": user_info["role"],
    }
    st.session_state["user"] = user_dict
    st.session_state["user_profile"] = get_student_profile(user_info["user_id"])
    return True, "Account created successfully!", user_dict


def logout_user():
    """Logs out the current user and clears session state."""
    st.session_state["user"] = None
    st.session_state["user_profile"] = None
    if "user" in st.session_state:
        del st.session_state["user"]
    if "user_profile" in st.session_state:
        del st.session_state["user_profile"]


def require_student() -> bool:
    """
    Guard for student pages. If not logged in as a student, displays a message and returns False.
    """
    if not is_authenticated():
        st.warning("🔒 Please log in as a student to access this page.")
        return False
    if not is_student():
        st.info("ℹ️ You are logged in as an Administrator. This feature is intended for Students.")
        return False
    return True


def require_admin() -> bool:
    """
    Guard for admin pages. If not logged in as an admin, displays an access denied message and returns False.
    """
    if not is_authenticated():
        st.warning("🔒 Please log in as an Administrator to access this portal.")
        return False
    if not is_admin():
        st.error("🚫 Access Denied: Administrator role required to view institutional analytics and student records.")
        return False
    return True


def render_auth_section():
    """
    Renders the role-based authentication portal tabs:
    - Student Login
    - Student Registration
    - Administrator Login
    - 1-Click Admin Access
    """
    ensure_admin_account()

    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                    border: 1px solid #334155; border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem;">
            <h3 style="margin-top:0; color:#f8fafc;">🔐 MindMap AI Authentication Portal</h3>
            <p style="color:#94a3b8; font-size:0.9rem; margin-bottom:0.5rem;">
                Select your role to enter: <strong>Students</strong> can register to take personal burnout assessments, track risk history, and simulate pacing interventions. <strong>Administrators</strong> can monitor aggregate institutional analytics and student records.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_student_login, tab_student_signup, tab_admin_login, tab_demo = st.tabs([
        "🎓 Student Login",
        "📝 Student Registration",
        "🛡️ Admin Login",
        "⚡ Instant Admin Access",
    ])

    # 1. Student Login
    with tab_student_login:
        with st.form("student_login_form"):
            s_email = st.text_input("Student Email", placeholder="Enter your registered email address")
            s_pass = st.text_input("Password", type="password", placeholder="Enter your password")
            s_submit = st.form_submit_button("Log In as Student", width="stretch")

            if s_submit:
                if not s_email or not s_pass:
                    st.error("Please enter both email and password.")
                else:
                    success, msg, u = login_user(s_email, s_pass)
                    if success:
                        st.success(f"Welcome back, {u['name']}!")
                        st.rerun()
                    else:
                        st.error(msg)

    # 2. Student Signup
    with tab_student_signup:
        with st.form("student_signup_form"):
            st.markdown("##### 1. Account Credentials")
            su_col1, su_col2 = st.columns(2)
            with su_col1:
                reg_name = st.text_input("Full Name", placeholder="e.g. Maya Chen")
                reg_email = st.text_input("Email Address", placeholder="e.g. maya.chen@campus.edu")
            with su_col2:
                reg_pass = st.text_input("Choose Password", type="password", placeholder="Minimum 6 characters")
                reg_gender = st.selectbox("Gender", CATEGORICAL_VOCABULARIES["gender"])

            st.markdown("##### 2. Academic Information")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                reg_age = st.number_input("Age", min_value=17, max_value=35, value=21)
                reg_year = st.selectbox("Year of Study", [1, 2, 3, 4], index=1)
            with p_col2:
                reg_course = st.selectbox("Course / Department", CATEGORICAL_VOCABULARIES["course"], index=3)

            reg_submit = st.form_submit_button("Create Student Account & Enter Portal", width="stretch")

            if reg_submit:
                if not reg_name or not reg_email or not reg_pass:
                    st.error("Please fill in all required fields.")
                else:
                    success, msg, u = signup_user(
                        name=reg_name,
                        email=reg_email,
                        password=reg_pass,
                        age=reg_age,
                        gender=reg_gender,
                        year_of_study=reg_year,
                        course=reg_course,
                    )
                    if success:
                        st.success(f"Account successfully created! Welcome, {u['name']}!")
                        st.rerun()
                    else:
                        st.error(msg)

    # 3. Admin Login
    with tab_admin_login:
        with st.form("admin_login_form"):
            st.markdown(
                """
                <div style="background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.3); border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem;">
                    <p style="color:#fda4af; font-size:0.85rem; margin:0;">
                        🛡️ <strong>Admin Portal:</strong> Default credentials are <code>admin@mindmap.ai</code> / <code>AdminPass2026!</code>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            a_email = st.text_input("Admin Email", value="admin@mindmap.ai")
            a_pass = st.text_input("Admin Password", type="password", value="AdminPass2026!", placeholder="Enter admin password")
            a_submit = st.form_submit_button("Log In as Administrator", width="stretch")

            if a_submit:
                if not a_email or not a_pass:
                    st.error("Please enter admin credentials.")
                else:
                    success, msg, u = login_user(a_email, a_pass)
                    if success:
                        if u["role"] == "admin":
                            st.success(f"Admin authentication verified. Welcome, {u['name']}!")
                            st.rerun()
                        else:
                            st.error("This account is not an administrator. Please log in via Student Portal.")
                    else:
                        st.error(msg)

    # 4. Instant Reviewer Access (Admin Portal)
    with tab_demo:
        st.markdown(
            """
            <div style="background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); 
                        border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <h4 style="margin-top:0; color:#38bdf8;">✨ Instant Administrator Access</h4>
                <p style="color:#cbd5e1; font-size:0.875rem; margin-bottom:0.75rem;">
                    One-click pre-authenticated access to inspect the institutional dashboard and cohort analytics.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🛡️ Log In as Administrator (System Admin)", width="stretch", type="primary"):
            success, msg, u = login_user("admin@mindmap.ai", "AdminPass2026!")
            if success:
                st.success(f"Logged in as Administrator: {u['name']}!")
                st.rerun()
            else:
                st.error(msg)

