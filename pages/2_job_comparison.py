"""
Job Comparison — select jobs from your results and compare them side-by-side.
"""

import plotly.graph_objects as go
import streamlit as st

from ui_helpers import SKILL_TAG_CSS, detect_title_column, render_sidebar, render_skill_tags

st.set_page_config(page_title="Job Comparison | ResuRank AI", page_icon="📄", layout="wide")
render_sidebar()
st.markdown(SKILL_TAG_CSS, unsafe_allow_html=True)

st.title("Job Comparison")
st.markdown("Select jobs from your results to compare them side-by-side.")

# ── Guard ──────────────────────────────────────────────────────────────────

if "ranking_results" not in st.session_state:
    st.info("Run a ranking on the **Ranked Results** page first.")
    st.stop()

results = st.session_state["ranking_results"]
df = st.session_state["ranking_df"]
skill_gaps = st.session_state["ranking_skill_gaps"]
title_col = detect_title_column(df)

# Build option labels
job_labels = []
for i, r in enumerate(results):
    idx = r["job_index"]
    title = df.iloc[idx][title_col] if title_col else "Untitled"
    job_labels.append(f"#{i+1} — {title}")

# ── Job selector ──────────────────────────────────────────────────

selected = st.multiselect(
    "Select 2-3 jobs to compare",
    options=range(len(job_labels)),
    format_func=lambda i: job_labels[i],
    max_selections=3,
)

if len(selected) < 2:
    st.info("Pick at least 2 jobs above to start comparing.")
    st.stop()

st.markdown("---")

# ── Score comparison chart ─────────────────────────────────────────────────

st.subheader("Score Comparison")

short_labels = []
for i in selected:
    idx = results[i]["job_index"]
    title = df.iloc[idx][title_col] if title_col else "Untitled"
    short_labels.append(title[:30] + ("..." if len(title) > 30 else ""))

fig = go.Figure()
fig.add_trace(go.Bar(
    name="TF-IDF",
    x=short_labels,
    y=[results[i]["tfidf_score"] for i in selected],
    marker_color="#4C78A8",
))
fig.add_trace(go.Bar(
    name="Semantic",
    x=short_labels,
    y=[results[i]["semantic_score"] for i in selected],
    marker_color="#F58518",
))
fig.add_trace(go.Bar(
    name="Final",
    x=short_labels,
    y=[results[i]["final_score"] for i in selected],
    marker_color="#54A24B",
))
fig.update_layout(
    barmode="group",
    margin=dict(t=20, b=40, l=40, r=20),
    height=350,
    yaxis_title="Score",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
)
st.plotly_chart(fig, use_container_width=True)

# ── Side-by-side detail cards ──────────────────────────────────────────────

st.markdown("---")
st.subheader("Detail Comparison")

cols = st.columns(len(selected))

for col, sel_idx in zip(cols, selected):
    r = results[sel_idx]
    gap = skill_gaps[sel_idx]
    idx = r["job_index"]
    title = df.iloc[idx][title_col] if title_col else "Untitled"
    desc = df.iloc[idx]["job_description_clean"]

    with col:
        st.markdown(f"**{title}**")

        # Scores
        st.metric("Final Score", f"{r['final_score']:.3f}")
        st.caption(f"TF-IDF: {r['tfidf_score']:.3f}  |  Semantic: {r['semantic_score']:.3f}")

        # Skill match
        match_pct = gap["match_percentage"]
        st.progress(match_pct / 100, text=f"Skill Match: {match_pct:.0f}%")

        # Skills
        if gap["matching_skills"]:
            st.markdown("**Matching**")
            st.markdown(render_skill_tags(gap["matching_skills"], "green"), unsafe_allow_html=True)
        if gap["missing_skills"]:
            st.markdown("**Missing**")
            st.markdown(render_skill_tags(gap["missing_skills"], "red"), unsafe_allow_html=True)

        # Description preview
        with st.expander("Job Description"):
            st.caption(desc[:600] + ("..." if len(desc) > 600 else ""))

# ── Skill overlap analysis ─────────────────────────────────────────────────

st.markdown("---")
st.subheader("Skill Overlap")

selected_gaps = [skill_gaps[i] for i in selected]
all_matching = [set(g["matching_skills"]) for g in selected_gaps]
all_missing = [set(g["missing_skills"]) for g in selected_gaps]

# Skills matching in all selected jobs
common_matching = set.intersection(*all_matching) if all_matching else set()
# Skills missing from all selected jobs
common_missing = set.intersection(*all_missing) if all_missing else set()

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Skills you match in all selected jobs**")
    if common_matching:
        st.markdown(render_skill_tags(common_matching, "green"), unsafe_allow_html=True)
    else:
        st.caption("No skills in common across all selected jobs.")

with col_b:
    st.markdown("**Skills missing from all selected jobs**")
    if common_missing:
        st.markdown(render_skill_tags(common_missing, "red"), unsafe_allow_html=True)
    else:
        st.caption("No universally missing skills.")
