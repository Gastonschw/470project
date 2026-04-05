"""
test_core.py
End-to-end test of the ResuRank AI core algorithm.

Demonstrates:
  1. Loading and preprocessing job postings
  2. Parsing a sample resume
  3. Ranking jobs with TF-IDF, Semantic, and Hybrid rankers
  4. Extracting skills and identifying skill gaps
  5. Evaluating with precision@k metrics

Usage:
    python test_core.py --data <path_to_csv> --resume <path_to_resume>
    python test_core.py --data <path_to_csv> --demo   (uses built-in sample resume)
"""

import argparse
import sys
import time

from data_loader import load_job_data, clean_text
from resume_parser import parse_resume
from ranker import TFIDFRanker, SemanticRanker, HybridRanker
from skill_extractor import extract_skills, analyze_skill_gap
from evaluation import evaluate_ranker

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


def display_results(results, df, resume_text, ranker_name):
    """Display ranked job results with skill gap analysis."""
    print(f"\nTop 5 matches ({ranker_name}):")
    print("-" * 50)

    # Determine title column
    title_col = None
    for col in ["job_title", "title", "jobtitle"]:
        if col in df.columns:
            title_col = col
            break

    resume_skills = extract_skills(resume_text)

    for i, result in enumerate(results[:5]):
        if isinstance(result, dict):
            idx = result["job_index"]
            score = result["final_score"]
            extra = f" (TF-IDF: {result['tfidf_score']:.3f}, Semantic: {result['semantic_score']:.3f})"
        else:
            idx, score = result
            extra = ""

        title = df.iloc[idx][title_col] if title_col else "N/A"
        desc_preview = df.iloc[idx]["job_description_clean"][:100]

        # Skill gap for this job
        job_skills = extract_skills(df.iloc[idx]["job_description_clean"])
        gap = analyze_skill_gap(resume_skills, job_skills)

        print(f"\n  #{i+1} [Score: {score:.4f}]{extra}")
        print(f"  Title: {title}")
        print(f"  Description: {desc_preview}...")
        print(f"  Skill Match: {gap['match_percentage']}% ({len(gap['matching_skills'])}/{gap['job_skill_count']})")
        if gap["matching_skills"]:
            print(f"  Matching: {', '.join(gap['matching_skills'][:8])}")
        if gap["missing_skills"]:
            print(f"  Missing:  {', '.join(gap['missing_skills'][:8])}")


def main():
    parser = argparse.ArgumentParser(description="ResuRank AI - Core Algorithm Test")
    parser.add_argument("--data", required=True, help="Path to CareerBuilder CSV file")
    parser.add_argument("--resume", help="Path to resume file (PDF or DOCX)")
    parser.add_argument("--demo", action="store_true", help="Use built-in sample resume")
    parser.add_argument("--sample", type=int, default=500,
                        help="Number of job postings to sample for faster testing (default: 500)")
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

    # ---- TF-IDF Ranking ----
    print_divider("4. TF-IDF Ranking (Lexical Baseline)")
    t0 = time.time()
    tfidf_ranker = TFIDFRanker()
    tfidf_ranker.fit(descriptions)
    tfidf_results = tfidf_ranker.rank(resume_clean, top_k=args.top_k)
    tfidf_time = time.time() - t0
    print(f"TF-IDF ranking completed in {tfidf_time:.2f}s")
    display_results(tfidf_results, df, resume_text, "TF-IDF")

    # ---- Semantic Ranking ----
    print_divider("5. Semantic Ranking (Sentence Transformers)")
    t0 = time.time()
    semantic_ranker = SemanticRanker()
    semantic_ranker.fit(descriptions)
    semantic_results = semantic_ranker.rank(resume_clean, top_k=args.top_k)
    semantic_time = time.time() - t0
    print(f"Semantic ranking completed in {semantic_time:.2f}s")
    display_results(semantic_results, df, resume_text, "Semantic")

    # ---- Hybrid Ranking ----
    print_divider("6. Hybrid Ranking (TF-IDF + Semantic)")
    t0 = time.time()
    hybrid_ranker = HybridRanker(alpha=args.alpha)
    hybrid_ranker.fit(descriptions)
    hybrid_results = hybrid_ranker.rank(resume_clean, top_k=args.top_k)
    hybrid_time = time.time() - t0
    print(f"Hybrid ranking (alpha={args.alpha}) completed in {hybrid_time:.2f}s")
    display_results(hybrid_results, df, resume_text, "Hybrid")

    # ---- Evaluation Summary ----
    print_divider("7. Summary")
    print(f"Dataset:      {len(df)} job postings")
    print(f"Resume:       {len(resume_skills)} skills extracted")
    print(f"TF-IDF time:  {tfidf_time:.2f}s")
    print(f"Semantic time: {semantic_time:.2f}s")
    print(f"Hybrid time:  {hybrid_time:.2f}s")
    print(f"Alpha:        {args.alpha}")
    print(f"\nNote: For precision@k evaluation, generate ground truth labels")
    print(f"using evaluation.py's create_eval_dataset_template() function.")
    print(f"See README.md for instructions.")


if __name__ == "__main__":
    main()
