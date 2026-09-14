"""
MindMap AI - Streamlit UI Styling & Shared Components
Provides custom modern CSS, Light/Dark theme engine, KPI card builders,
risk badges, role-aware sidebars, and standardized disclaimers.
"""

import streamlit as st
from typing import Dict, Any, Optional
import config
from db.connection import check_db_status


def get_current_theme() -> str:
    """Returns the current active UI theme ('dark' or 'light')."""
    if "app_theme" not in st.session_state:
        st.session_state["app_theme"] = "dark"
    return st.session_state["app_theme"]


def set_current_theme(theme: str):
    """Sets the active UI theme."""
    st.session_state["app_theme"] = theme


def get_plotly_layout_defaults() -> Dict[str, Any]:
    """Returns Plotly layout parameters matching the active UI theme."""
    theme = get_current_theme()
    is_dark = theme == "dark"
    return {
        "template": "plotly_dark" if is_dark else "plotly_white",
        "font_color": "#f8fafc" if is_dark else "#0f172a",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
    }


def inject_custom_css():
    """
    Injects custom modern CSS for a premium, clean, professional data science UI.
    Dynamically adjusts for Dark Mode and Light Mode.
    """
    theme = get_current_theme()
    is_dark = theme == "dark"

    if is_dark:
        bg_main = "#0b0f19"
        bg_card = "#1e293b"
        bg_factor = "#0f172a"
        border_card = "#334155"
        border_hover = "#64748b"
        text_primary = "#f8fafc"
        text_secondary = "#94a3b8"
        text_muted = "#64748b"
        chip_student_bg = "rgba(30, 41, 59, 0.85)"
        chip_student_border = "#38bdf8"
        chip_admin_bg = "rgba(88, 28, 135, 0.2)"
        chip_admin_border = "#c084fc"
        disclaimer_bg = "rgba(15, 23, 42, 0.6)"
        disclaimer_border = "#334155"
        disclaimer_accent = "#38bdf8"
        header_gradient = "linear-gradient(135deg, #1e293b 0%, #0f172a 100%)"
        brand_color = "#38bdf8"
    else:
        bg_main = "#f8fafc"
        bg_card = "#ffffff"
        bg_factor = "#f1f5f9"
        border_card = "#e2e8f0"
        border_hover = "#94a3b8"
        text_primary = "#0f172a"
        text_secondary = "#475569"
        text_muted = "#94a3b8"
        chip_student_bg = "rgba(238, 242, 255, 0.9)"
        chip_student_border = "#0284c7"
        chip_admin_bg = "rgba(243, 232, 255, 0.9)"
        chip_admin_border = "#9333ea"
        disclaimer_bg = "#f1f5f9"
        disclaimer_border = "#cbd5e1"
        disclaimer_accent = "#0284c7"
        header_gradient = "linear-gradient(135deg, #e2e8f0 0%, #f8fafc 100%)"
        brand_color = "#0284c7"

    css_block = f"""
    <style>
    /* Modern Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* Global Brand Banner at Top */
    .brand-top-banner {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 0 1rem 0;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid {border_card};
    }}
    .brand-top-title {{
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        color: {brand_color};
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    .brand-top-subtitle {{
        font-size: 0.85rem;
        color: {text_secondary};
        font-weight: 500;
    }}

    /* Page Header Card */
    .main-header {{
        background: {header_gradient};
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border: 1px solid {border_card};
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }}
    .main-header h1 {{
        color: {text_primary};
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.025em;
    }}
    .main-header p {{
        color: {text_secondary};
        font-size: 0.95rem;
        margin: 0.35rem 0 0 0;
    }}

    /* KPI Summary Cards */
    .kpi-card {{
        background: {bg_card};
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        border: 1px solid {border_card};
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}
    .kpi-card:hover {{
        border-color: {border_hover};
        transform: translateY(-2px);
    }}
    .kpi-title {{
        color: {text_secondary};
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }}
    .kpi-value {{
        color: {text_primary};
        font-size: 1.85rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }}
    .kpi-subtitle {{
        color: {text_muted};
        font-size: 0.8rem;
    }}

    /* User Profile Pills */
    .user-profile-chip-student {{
        background: {chip_student_bg};
        border: 1px solid {chip_student_border};
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 1rem;
    }}
    .user-profile-chip-admin {{
        background: {chip_admin_bg};
        border: 1px solid {chip_admin_border};
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 1rem;
    }}
    .user-profile-name {{
        color: {text_primary};
        font-weight: 600;
        font-size: 0.95rem;
    }}
    .user-profile-detail {{
        color: {text_secondary};
        font-size: 0.8rem;
        margin-top: 0.2rem;
    }}

    /* Risk Badges */
    .risk-badge {{
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.025em;
    }}
    .risk-low {{
        background-color: rgba(34, 197, 94, 0.15);
        color: #16a34a;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }}
    .risk-moderate {{
        background-color: rgba(234, 179, 8, 0.15);
        color: #ca8a04;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }}
    .risk-high {{
        background-color: rgba(249, 115, 22, 0.15);
        color: #ea580c;
        border: 1px solid rgba(249, 115, 22, 0.3);
    }}
    .risk-very-high {{
        background-color: rgba(239, 68, 68, 0.15);
        color: #dc2626;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }}

    /* Contributing Factor & Recommendation Card */
    .factor-card {{
        background: {bg_factor};
        border: 1px solid {border_card};
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }}
    .factor-card.elevates {{
        border-left: 4px solid #ef4444;
    }}
    .factor-card.lowers {{
        border-left: 4px solid #22c55e;
    }}

    /* SWOT Box Styles */
    .swot-card {{
        background: {bg_card};
        border: 1px solid {border_card};
        border-radius: 10px;
        padding: 1.15rem;
        height: 100%;
    }}
    .swot-title {{
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }}

    /* Disclaimer Banner */
    .disclaimer-box {{
        background: {disclaimer_bg};
        border: 1px solid {disclaimer_border};
        border-left: 4px solid {disclaimer_accent};
        padding: 0.85rem 1.25rem;
        border-radius: 6px;
        font-size: 0.825rem;
        color: {text_secondary};
        margin-top: 1.5rem;
        line-height: 1.4;
    }}

    /* Database Status Pill */
    .db-status-pill {{
        font-size: 0.75rem;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        background: {bg_card};
        border: 1px solid {border_card};
        color: {text_secondary};
        display: inline-block;
        margin-bottom: 0.5rem;
    }}
    </style>
    """
    st.markdown(css_block, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, icon: str = "🧠"):
    """Renders prominent top branding hierarchy and standardized page header."""
    theme = get_current_theme()
    brand_color = "#38bdf8" if theme == "dark" else "#0284c7"
    st.markdown(
        f"""
        <div class="brand-top-banner">
            <div class="brand-top-title">🧠 MINDMAP AI</div>
            <div class="brand-top-subtitle">Student Burnout Intelligence Platform</div>
        </div>
        <div class="main-header">
            <h1>{icon} {title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Renders standard branding, theme toggle, role badge, authentication status, and system health in the sidebar."""
    with st.sidebar:
        st.markdown("### 🧠 **MINDMAP AI**")
        st.caption("Student Burnout Intelligence Platform")
        st.markdown("---")

        # Theme Switcher (Light / Dark Mode)
        current_theme = get_current_theme()
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            if st.button("🌙 Dark", key="theme_dark_btn", width="stretch", type="primary" if current_theme == "dark" else "secondary"):
                set_current_theme("dark")
                st.rerun()
        with t_col2:
            if st.button("☀ Light", key="theme_light_btn", width="stretch", type="primary" if current_theme == "light" else "secondary"):
                set_current_theme("light")
                st.rerun()

        st.markdown("---")

        from utils.auth import is_authenticated, get_current_user, get_current_profile, is_admin, is_student, logout_user

        if is_authenticated():
            user = get_current_user()
            if is_admin():
                st.markdown(
                    f"""
                    <div class="user-profile-chip-admin">
                        <div class="user-profile-name">🛡️ {user.get('name', 'Administrator')}</div>
                        <div class="user-profile-detail">🏢 Institutional Admin Portal</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                profile = get_current_profile() or {}
                full_name = profile.get("full_name", user.get("name", "Student"))
                course = profile.get("course", "Computer Science")
                year = profile.get("year_of_study", 1)

                st.markdown(
                    f"""
                    <div class="user-profile-chip-student">
                        <div class="user-profile-name">🎓 {full_name}</div>
                        <div class="user-profile-detail">📚 {course} · Year {year}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if st.button("🚪 Log Out", key="sidebar_logout_btn", width="stretch"):
                logout_user()
                st.rerun()
        else:
            st.markdown(
                """
                <div style="background: rgba(30, 41, 59, 0.4); border: 1px dashed #64748b; 
                            border-radius: 6px; padding: 0.6rem; margin-bottom: 0.8rem; font-size: 0.8rem; color: #94a3b8;">
                    👤 <strong>Guest Mode</strong> (Log in on the Home page to unlock full personalized access)
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Database connection indicator
        db_info = check_db_status()
        engine_label = "🟢 MySQL" if db_info["engine"] == "MySQL" else "🟡 SQLite Engine"
        st.markdown(
            f"""
            <div class="db-status-pill">
                <strong>Database:</strong> {engine_label}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("MindMap AI v2.0 · Personalized Multi-User Web App")


def render_disclaimer(custom_text: Optional[str] = None):
    """Renders standardized non-medical disclaimer banner."""
    text = custom_text or config.NON_MEDICAL_DISCLAIMER
    st.markdown(
        f"""
        <div class="disclaimer-box">
            <strong>⚠️ Non-Medical Disclaimer:</strong> {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_research_disclaimer():
    """Renders specific disclaimer for published external research context."""
    st.markdown(
        """
        <div class="disclaimer-box" style="border-left-color: #a855f7;">
            <strong>📚 Research Context Note:</strong> Research statistics shown here are from published peer-reviewed studies and are provided for contextual scientific comparison only. They do not represent MindMap AI users.
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_risk_badge_html(category: str) -> str:
    """Returns HTML snippet for risk category badge."""
    cat_lower = category.lower().replace(" ", "-")
    return f'<span class="risk-badge risk-{cat_lower}">{category} Risk</span>'
