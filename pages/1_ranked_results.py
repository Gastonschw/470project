"""
Ranked Results — upload a resume and see top matching jobs.
"""

import streamlit as st
from data_loader import clean_text
from ranker import HybridRanker
from skill_extractor import extract_skills, analyze_skill_gap
from ui_helpers import (
    SKILL_TAG_CSS,
    detect_title_column,
    get_cached_data,
    get_fitted_rankers,
    make_desc_hash,
    parse_uploaded_resume,
    render_sidebar,
    render_skill_tags,
)

st.set_page_config(page_title="Ranked Results | ResuRank AI", page_icon="📄", layout="wide")
render_sidebar()

# Demo resume (same as test_core.py)
SAMPLE_RESUME = """
Gaston Schweitzer
Computer Science Student | Texas A&M University

EDUCATION
Bachelor of Science in Computer Science, Texas A&M University, Expected May 2026
Relevant Coursework: Operating Systems, Information Retrieval, Software Engineering,
Data Structures, Algorithms, Systems Programming

SKILLS
Programming Languages: Python, Java, C++, JavaScript, SQL, HTML, CSS
Frameworks: React, Flask, Django, scikit-learn, pandas, NumPy
Tools: Git, GitHub, Docker, Linux, VS Code
Concepts: Machine Learning, Natural Language Processing, Data Analysis,
Agile Development, REST API, Object-Oriented Programming

EXPERIENCE
Software Engineering Intern
- Developed REST APIs using Python and Flask
- Built data pipelines for processing and analyzing large datasets
- Collaborated with cross-functional teams using Agile/Scrum methodology
- Wrote unit tests and performed code reviews

PROJECTS
- AI-Powered Code Review Agent: Built a tool that analyzes code quality using LLMs
- CS2 Match Prediction Model: Statistical model using Bradley-Terry MLE
- Information Retrieval System: Implemented TF-IDF, BM25, and cosine similarity search
"""

st.markdown(SKILL_TAG_CSS, unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────

st.title("Ranked Results")
st.markdown("Upload your resume to find the best matching jobs from our database.")

# ── Resume input ───────────────────────────────────────────────────────────

use_demo = st.checkbox("Use demo resume", value=True)

if use_demo:
    resume_text = SAMPLE_RESUME
    st.info("Using the built-in demo resume.")
else:
    uploaded = st.file_uploader("Upload your resume", type=["pdf", "docx"])
    if uploaded:
        with st.spinner("Parsing resume..."):
            resume_text = parse_uploaded_resume(uploaded)
    else:
        resume_text = None

# ── Advanced settings ──────────────────────────────────────────────────────

with st.expander("Advanced Settings"):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        sample_size = st.number_input(
            "Job sample size", min_value=50, max_value=30000,
            value=st.session_state.get("sample_size", 500), step=50,
        )
        st.session_state["sample_size"] = sample_size
    with col_b:
        top_k = st.number_input(
            "Number of results", min_value=5, max_value=50,
            value=st.session_state.get("top_k", 10),
        )
        st.session_state["top_k"] = top_k
    with col_c:
        alpha = st.slider(
            "TF-IDF vs Semantic blend",
            min_value=0.0, max_value=1.0,
            value=st.session_state.get("alpha", 0.5), step=0.05,
            help="0 = pure semantic, 1 = pure TF-IDF",
        )
        st.session_state["alpha"] = alpha

# ── Run ranking ────────────────────────────────────────────────────────────

if resume_text and st.button("Find Matching Jobs", type="primary", use_container_width=True):
    data_path = st.session_state.get("data_path", "")
    sample_size = st.session_state.get("sample_size", 500)
    top_k = st.session_state.get("top_k", 10)
    alpha = st.session_state.get("alpha", 0.5)

    # Load data
    df = get_cached_data(data_path, sample_size)
    descriptions = df["job_description_clean"].tolist()
    title_col = detect_title_column(df)

    # Fit rankers (cached)
    desc_hash = make_desc_hash(descriptions)
    tfidf_ranker, semantic_ranker = get_fitted_rankers(descriptions, desc_hash)

    # Rank
    with st.spinner("Ranking jobs..."):
        hybrid = HybridRanker(
            alpha=alpha,
            tfidf_ranker=tfidf_ranker,
            semantic_ranker=semantic_ranker,
        )
        hybrid.fit(descriptions)
        resume_clean = clean_text(resume_text)
        results = hybrid.rank(resume_clean, top_k=top_k)

    # Extract resume skills once
    resume_skills = extract_skills(resume_text)

    # Compute skill gaps for each result
    skill_gaps = []
    for r in results:
        idx = r["job_index"]
        job_desc = df.iloc[idx]["job_description_clean"]
        job_skills = extract_skills(job_desc)
        gap = analyze_skill_gap(resume_skills, job_skills)
        skill_gaps.append(gap)

    # Store in session state for other pages
    st.session_state["ranking_results"] = results
    st.session_state["ranking_df"] = df
    st.session_state["ranking_skill_gaps"] = skill_gaps
    st.session_state["resume_text"] = resume_text
    st.session_state["resume_skills"] = resume_skills

    # ── Display results ────────────────────────────────────────────────────

    st.markdown("---")
    st.subheader(f"Top {len(results)} Matches")

    for i, (result, gap) in enumerate(zip(results, skill_gaps)):
        idx = result["job_index"]
        title = df.iloc[idx][title_col] if title_col else "Untitled"
        desc = df.iloc[idx]["job_description_clean"]
        score = result["final_score"]
        match_pct = gap["match_percentage"]

        with st.expander(f"#{i+1}  {title}  —  Score: {score:.3f}", expanded=(i < 3)):
            # Top row: scores + skill match on one line
            c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
            c1.metric("Final Score", f"{score:.3f}")
            c2.metric("TF-IDF", f"{result['tfidf_score']:.3f}")
            c3.metric("Semantic", f"{result['semantic_score']:.3f}")
            c4.metric("Skill Match", f"{match_pct:.0f}%")

            # Skill tags
            skill_html = ""
            if gap["matching_skills"]:
                skill_html += f'<p style="margin-bottom:4px;"><b>Matching:</b> {render_skill_tags(gap["matching_skills"], "green")}</p>'
            if gap["missing_skills"]:
                skill_html += f'<p style="margin-bottom:4px;"><b>Missing:</b> {render_skill_tags(gap["missing_skills"], "red")}</p>'
            if gap["extra_skills"]:
                skill_html += f'<p style="margin-bottom:4px;"><b>Extra:</b> {render_skill_tags(gap["extra_skills"], "gray")}</p>'
            if skill_html:
                st.markdown(skill_html, unsafe_allow_html=True)

            # Description
            st.caption(desc[:400] + ("..." if len(desc) > 400 else ""))

elif not resume_text and not use_demo:
    st.info("Upload a resume above to get started.")
