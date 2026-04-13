"""
ResuRank AI — Streamlit UI
"""

import streamlit as st

from ui_helpers import render_sidebar

st.set_page_config(
    page_title="ResuRank AI",
    page_icon="📄",
    layout="wide",
)
render_sidebar()

# ── Defaults ─────────────────────────────────────────────

DEFAULT_DATA_PATH = (
    "marketing_sample_for_careerbuilder_usa-careerbuilder_job_listing"
    "__20200401_20200630__30k_data.ldjson"
)

st.session_state.setdefault("data_path", DEFAULT_DATA_PATH)
st.session_state.setdefault("sample_size", 500)
st.session_state.setdefault("top_k", 10)
st.session_state.setdefault("alpha", 0.5)

# ── Landing page ───────────────────────────────────────────────────────────

# Card styling for dark/light theme compatibility
st.markdown(
    """
    <style>
    .feature-card {
        border: 1px solid rgba(150, 150, 150, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        height: 100%;
        transition: border-color 0.2s;
    }
    .feature-card:hover {
        border-color: rgba(150, 150, 150, 0.5);
    }
    .feature-card h3 {
        margin-bottom: 0.5rem;
    }
    .feature-card p {
        color: #999;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .step-number {
        display: inline-block;
        width: 28px;
        height: 28px;
        line-height: 28px;
        border-radius: 50%;
        background: rgba(150, 150, 150, 0.15);
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="text-align: center; padding: 3rem 0 1.5rem 0;">
        <h1 style="font-size: 2.8rem; margin-bottom: 0.3rem;">ResuRank AI</h1>
        <p style="font-size: 1.15rem; color: #999; margin-top: 0;">
            Find your best-fit jobs using intelligent resume matching
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown(
        """
        <div class="feature-card">
            <span class="step-number">1</span>
            <h3>Ranked Results</h3>
            <p>
                Upload your resume and instantly see the top matching jobs,
                ranked by a hybrid blend of keyword and semantic similarity.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="feature-card">
            <span class="step-number">2</span>
            <h3>Job Comparison</h3>
            <p>
                Select your favorite matches and compare them side-by-side
                on scores, skill requirements, and descriptions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="feature-card">
            <span class="step-number">3</span>
            <h3>Skill Analysis</h3>
            <p>
                See your resume's skill profile, discover gaps for each job,
                and learn what employers are looking for most.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <p style="color: #999; font-size: 1rem;">
            Select <b>Ranked Results</b> in the sidebar to get started.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
