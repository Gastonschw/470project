"""
Skill Analysis — resume insights and skill gap deep-dive.
"""

from collections import Counter

import plotly.graph_objects as go
import streamlit as st

from skill_extractor import SKILL_CATEGORIES
from ui_helpers import SKILL_TAG_CSS, detect_title_column, render_sidebar, render_skill_tags

st.set_page_config(page_title="Skill Analysis | ResuRank AI", page_icon="📄", layout="wide")
render_sidebar()
st.markdown(SKILL_TAG_CSS, unsafe_allow_html=True)

st.title("Skill Analysis")
st.markdown("Understand your resume's strengths and see how you match against top jobs.")

# ── Guard ──────────────────────────────────────────────────────────────────

if "ranking_results" not in st.session_state:
    st.info("Run a ranking on the **Ranked Results** page first.")
    st.stop()

results = st.session_state["ranking_results"]
df = st.session_state["ranking_df"]
skill_gaps = st.session_state["ranking_skill_gaps"]
resume_skills = st.session_state.get("resume_skills", set())
title_col = detect_title_column(df)

# ── Resume Insights ────────────────────────────────────────────────────────

st.subheader("Your Resume Profile")

# Build category counts for resume skills
category_data = {}
for cat_name, cat_skills in SKILL_CATEGORIES.items():
    matched = resume_skills & cat_skills
    if matched:
        category_data[cat_name] = sorted(matched)

# Overview metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Skills Detected", len(resume_skills))
col2.metric("Skill Categories", len(category_data))
avg_match = sum(g["match_percentage"] for g in skill_gaps) / len(skill_gaps) if skill_gaps else 0
col3.metric("Avg Job Match", f"{avg_match:.0f}%")

# Category breakdown with donut + tags
if category_data:
    cat_col, tag_col = st.columns([1, 2])

    with cat_col:
        fig = go.Figure(data=[go.Pie(
            labels=list(category_data.keys()),
            values=[len(v) for v in category_data.values()],
            hole=0.5,
            textinfo="label+value",
            textfont_size=13,
            marker=dict(colors=["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]),
        )])
        fig.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tag_col:
        for cat_name, skills in category_data.items():
            st.markdown(f"**{cat_name}**")
            st.markdown(render_skill_tags(skills, "green"), unsafe_allow_html=True)

# ── Job-Specific Skill Gap ─────────────────────────────────────────────────

st.markdown("---")
st.subheader("Job Skill Gap")

options = []
for i, r in enumerate(results):
    idx = r["job_index"]
    title = df.iloc[idx][title_col] if title_col else "Untitled"
    options.append(f"#{i+1} — {title} (score: {r['final_score']:.3f})")

selected_idx = st.selectbox("Select a job to analyze", range(len(options)), format_func=lambda i: options[i])
gap = skill_gaps[selected_idx]

col_chart, col_details = st.columns([1, 2])

with col_chart:
    matching_count = len(gap["matching_skills"])
    missing_count = len(gap["missing_skills"])

    fig = go.Figure(data=[go.Pie(
        labels=["Matching", "Missing"],
        values=[matching_count, missing_count],
        hole=0.55,
        marker=dict(colors=["#28a745", "#dc3545"]),
        textinfo="label+value",
        textfont_size=14,
    )])
    fig.update_layout(
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        height=280,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_details:
    delta = gap["match_percentage"] - avg_match
    st.metric(
        "Skill Match",
        f"{gap['match_percentage']:.0f}%",
        delta=f"{delta:+.1f}% vs avg",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Matching**")
        if gap["matching_skills"]:
            st.markdown(render_skill_tags(gap["matching_skills"], "green"), unsafe_allow_html=True)
        else:
            st.caption("None")
    with c2:
        st.markdown("**Missing**")
        if gap["missing_skills"]:
            st.markdown(render_skill_tags(gap["missing_skills"], "red"), unsafe_allow_html=True)
        else:
            st.caption("None")
    with c3:
        st.markdown("**Your Extra Skills**")
        if gap["extra_skills"]:
            st.markdown(render_skill_tags(gap["extra_skills"], "gray"), unsafe_allow_html=True)
        else:
            st.caption("None")

# ── Aggregate insights ─────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Market Trends")
st.markdown("What employers want most across your top matches.")

missing_counter = Counter()
matching_counter = Counter()
for g in skill_gaps:
    missing_counter.update(g["missing_skills"])
    matching_counter.update(g["matching_skills"])

col_miss, col_match = st.columns(2)

with col_miss:
    st.markdown("**Most Commonly Missing**")
    top_missing = missing_counter.most_common(10)
    if top_missing:
        labels, counts = zip(*reversed(top_missing))
        fig = go.Figure(go.Bar(
            x=list(counts), y=list(labels),
            orientation="h",
            marker_color="#dc3545",
        ))
        fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=300,
            xaxis_title="Frequency",
            yaxis=dict(automargin=True),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No missing skills found.")

with col_match:
    st.markdown("**Your Most Valuable Skills**")
    top_matching = matching_counter.most_common(10)
    if top_matching:
        labels, counts = zip(*reversed(top_matching))
        fig = go.Figure(go.Bar(
            x=list(counts), y=list(labels),
            orientation="h",
            marker_color="#28a745",
        ))
        fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=300,
            xaxis_title="Frequency",
            yaxis=dict(automargin=True),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No matching skills found.")
