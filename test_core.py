"""
test_core.py
End-to-end test of the ResuRank AI core algorithm.

Demonstrates:
  1. Loading and preprocessing job postings
  2. Parsing a sample resume
  3. Ranking jobs with the Hybrid ranker (TF-IDF + Semantic)
  4. Extracting skills and identifying skill gaps

Usage:
    python test_core.py --data <path_to_ldjson> --resume <path_to_resume>
    python test_core.py --data <path_to_ldjson> --demo   (uses built-in sample resume)
"""

import argparse
import os
import sys
import time
import warnings

# Suppress HF Hub download warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", message=".*unauthenticated.*HF Hub.*")

from data_loader import load_job_data, clean_text
from resume_parser import parse_resume
from ranker import HybridRanker
from skill_extractor import extract_skills, analyze_skill_gap

# Sample resume for demo/testing purposes
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


def print_divider(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def display_results(results, df, resume_text):
    """Display ranked job results with skill gap analysis."""
    print(f"\nTop 5 matches (Hybrid):")
    print("-" * 50)

    # Determine title column
    title_col = None
    for col in ["job_title", "title", "jobtitle"]:
        if col in df.columns:
            title_col = col
            break

    resume_skills = extract_skills(resume_text)

    for i, result in enumerate(results[:5]):
        idx = result["job_index"]
        score = result["final_score"]

        title = df.iloc[idx][title_col] if title_col else "N/A"
        desc_preview = df.iloc[idx]["job_description_clean"][:100]

        # Skill gap for this job
        job_skills = extract_skills(df.iloc[idx]["job_description_clean"])
        gap = analyze_skill_gap(resume_skills, job_skills)

        print(f"\n  #{i+1} [Score: {score:.4f}] (TF-IDF: {result['tfidf_score']:.3f}, Semantic: {result['semantic_score']:.3f})")
        print(f"  Title: {title}")
        print(f"  Description: {desc_preview}...")
        print(f"  Skill Match: {gap['match_percentage']}% ({len(gap['matching_skills'])}/{gap['job_skill_count']})")
        if gap["matching_skills"]:
            print(f"  Matching: {', '.join(gap['matching_skills'][:8])}")
        if gap["missing_skills"]:
            print(f"  Missing:  {', '.join(gap['missing_skills'][:8])}")


def main():
    parser = argparse.ArgumentParser(description="ResuRank AI - Core Algorithm Test")
    parser.add_argument("--data", required=True, help="Path to CareerBuilder LDJSON file")
    parser.add_argument("--resume", help="Path to resume file (PDF or DOCX)")
    parser.add_argument("--demo", action="store_true", help="Use built-in sample resume")
    parser.add_argument("--sample", type=int, default=500,
                        help="Number of job postings to sample (default: 500)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top results to return (default: 10)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Hybrid blend weight: 0=pure semantic, 1=pure TF-IDF (default: 0.5)")
    args = parser.parse_args()

    # ---- Load Resume ----
    print_divider("1. Loading Resume")
    if args.resume:
        resume_text = parse_resume(args.resume)
        print(f"Parsed resume: {len(resume_text.split())} words")
    elif args.demo:
        resume_text = SAMPLE_RESUME
        print(f"Using sample resume: {len(resume_text.split())} words")
    else:
        print("Error: Provide --resume <file> or --demo")
        sys.exit(1)

    resume_clean = clean_text(resume_text)

    # ---- Extract Resume Skills ----
    print_divider("2. Resume Skill Extraction")
    resume_skills = extract_skills(resume_text)
    print(f"Found {len(resume_skills)} skills in resume:")
    print(f"  {', '.join(sorted(resume_skills))}")

    # ---- Load Job Data ----
    print_divider("3. Loading Job Postings")
    df = load_job_data(args.data, sample_size=args.sample)
    descriptions = df["job_description_clean"].tolist()

    # ---- Hybrid Ranking ----
    print_divider("4. Hybrid Ranking (TF-IDF + Semantic)")
    t0 = time.time()
    hybrid_ranker = HybridRanker(alpha=args.alpha)
    hybrid_ranker.fit(descriptions)
    hybrid_results = hybrid_ranker.rank(resume_clean, top_k=args.top_k)
    hybrid_time = time.time() - t0
    print(f"Hybrid ranking (alpha={args.alpha}) completed in {hybrid_time:.2f}s")
    display_results(hybrid_results, df, resume_text)

    # ---- Summary ----
    print_divider("5. Summary")
    print(f"Dataset:      {len(df)} job postings")
    print(f"Resume:       {len(resume_skills)} skills extracted")
    print(f"Ranking time: {hybrid_time:.2f}s")
    print(f"Alpha:        {args.alpha}")


if __name__ == "__main__":
    main()
