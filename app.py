"""
ResuRank AI — Streamlit UI
"""

import streamlit as st

st.set_page_config(
    page_title="ResuRank AI",
    page_icon="📄",
    layout="wide",
)

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

st.markdown(
    """
    <div style="text-align: center; padding: 2rem 0 1rem 0;">
        <h1 style="font-size: 2.8rem; margin-bottom: 0.2rem;">ResuRank AI</h1>
        <p style="font-size: 1.2rem; color: #666; margin-top: 0;">
            Find your best-fit jobs using intelligent resume matching
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem;">
            <h3>Ranked Results</h3>
            <p style="color: #555;">
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
        <div style="text-align: center; padding: 1rem;">
            <h3>Ranker Comparison</h3>
            <p style="color: #555;">
                See how TF-IDF, Semantic, and Hybrid rankers stack up
                with Precision, Recall, and Average Precision metrics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem;">
            <h3>Skill Analysis</h3>
            <p style="color: #555;">
                Discover which skills you're matching, which you're missing,
                and what employers are looking for most.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
