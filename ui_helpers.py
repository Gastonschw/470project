"""
ui_helpers.py
Shared utilities for the Streamlit UI: caching wrappers, display helpers etc.
"""

import hashlib
import os
import tempfile

import streamlit as st

from data_loader import load_job_data
from ranker import TFIDFRanker, SemanticRanker
from resume_parser import parse_resume


# ── Resume parsing wrapper ─────────────────────────────────────────────────

def parse_uploaded_resume(uploaded_file) -> str:
    """Parse a Streamlit UploadedFile (PDF/DOCX) into text."""
    suffix = "." + uploaded_file.name.rsplit(".", 1)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        text = parse_resume(tmp_path)
    finally:
        os.unlink(tmp_path)
    return text


# ── Cached data loading ────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading job data...")
def get_cached_data(data_path: str, sample_size: int = None):
    """Load and cache the job postings DataFrame."""
    return load_job_data(data_path, sample_size=sample_size)


# ── Cached ranker fitting ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="Fitting rankers (this may take a moment)...")
def get_fitted_rankers(_descriptions: list[str], desc_hash: str):
    """
    Fit and cache TF-IDF and Semantic rankers.
    """
    tfidf = TFIDFRanker()
    tfidf.fit(_descriptions)
    semantic = SemanticRanker()
    semantic.fit(_descriptions)
    return tfidf, semantic


def make_desc_hash(descriptions: list[str]) -> str:
    """Create a stable hash string from a list of descriptions for cache keying."""
    content = str(len(descriptions)) + descriptions[0][:200] + descriptions[-1][:200]
    return hashlib.md5(content.encode()).hexdigest()


# ── Column detection ───────────────────────────────────────────────────────

def detect_title_column(df) -> str | None:
    """Find the job title column name in the DataFrame."""
    for col in ("job_title", "title", "jobtitle"):
        if col in df.columns:
            return col
    return None


# ── Skill tag rendering ───────────────────────────────────────────────────

SKILL_TAG_CSS = """
<style>
.skill-tag {
    display: inline-block;
    padding: 3px 12px;
    margin: 3px 4px;
    border-radius: 14px;
    font-size: 0.85em;
    font-weight: 500;
    letter-spacing: 0.01em;
}
.skill-green { background-color: #d4edda; color: #155724; }
.skill-red   { background-color: #f8d7da; color: #721c24; }
.skill-gray  { background-color: #e9ecef; color: #495057; }
</style>
"""


def render_skill_tags(skills, color: str) -> str:
    """
    Return an HTML string of colored skill badges.

    Args:
        skills: Iterable of skill names.
        color: One of 'green', 'red', 'gray'.
    """
    css_class = f"skill-{color}"
    tags = "".join(
        f'<span class="skill-tag {css_class}">{skill}</span>'
        for skill in sorted(skills)
    )
    return tags


# ── Shared sidebar ────────────────────────────────────────────────────────

def render_sidebar():
    """Render consistent sidebar across all pages."""
    st.sidebar.markdown(
        """
        <div style="padding: 0 0.5rem;">
            <p style="color: #999; font-size: 0.8rem; line-height: 1.6;">
                <b>How to use</b><br>
                1. Upload your resume on <b>Ranked Results</b><br>
                2. Compare top matches on <b>Job Comparison</b><br>
                3. Explore skill gaps on <b>Skill Analysis</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
