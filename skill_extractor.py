"""
skill_extractor.py
Extracts skills from resumes and job descriptions and identifies skill gaps.

Uses a predefined skill taxonomy + simple NLP matching.
For a more robust approach, you could use spaCy NER or an LLM.
"""

import re

# Common technical and professional skills taxonomy
# Expand this list based on your dataset's domain
SKILL_TAXONOMY = {
    # Programming languages
    "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "php",
    "swift", "kotlin", "go", "rust", "scala", "r", "matlab", "sql", "html",
    "css", "bash", "shell", "perl",

    # Frameworks & libraries
    "react", "angular", "vue", "django", "flask", "spring", "node.js", "express",
    "rails", "laravel", ".net", "tensorflow", "pytorch", "keras", "pandas",
    "numpy", "scikit-learn", "jquery", "bootstrap", "tailwind",

    # Tools & platforms
    "git", "github", "docker", "kubernetes", "aws", "azure", "gcp",
    "jenkins", "terraform", "ansible", "linux", "windows", "jira", "confluence",
    "tableau", "power bi", "excel", "salesforce", "sap",

    # Data & databases
    "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "dynamodb",
    "sqlite", "oracle", "sql server", "hadoop", "spark", "kafka", "airflow",

    # Concepts & methodologies
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "data analysis", "data science", "data engineering",
    "devops", "ci/cd", "agile", "scrum", "rest api", "microservices",
    "object-oriented", "test-driven", "cloud computing",

    # Soft skills & business
    "project management", "leadership", "communication", "problem solving",
    "teamwork", "critical thinking", "time management", "customer service",
    "sales", "marketing", "accounting", "financial analysis", "budgeting",

    # Certifications & standards
    "pmp", "aws certified", "azure certified", "cissp", "comptia",
    "six sigma", "lean", "itil",
}


def extract_skills(text: str, taxonomy: set = None) -> set[str]:
    """
    Extract skills from text using taxonomy matching.

    Args:
        text: Input text (resume or job description).
        taxonomy: Set of known skills to match against.

    Returns:
        Set of matched skills found in the text.
    """
    if taxonomy is None:
        taxonomy = SKILL_TAXONOMY

    text_lower = text.lower()
    found_skills = set()

    for skill in taxonomy:
        # Use word boundary matching for short skills to avoid false positives
        if len(skill) <= 3:
            pattern = r"\b" + re.escape(skill) + r"\b"
            if re.search(pattern, text_lower):
                found_skills.add(skill)
        else:
            if skill in text_lower:
                found_skills.add(skill)

    return found_skills


def analyze_skill_gap(resume_skills: set[str], job_skills: set[str]) -> dict:
    """
    Compare resume skills against job requirements.

    Args:
        resume_skills: Skills extracted from the resume.
        job_skills: Skills extracted from the job description.

    Returns:
        Dict with matching skills, missing skills, extra skills, and match percentage.
    """
    matching = resume_skills & job_skills
    missing = job_skills - resume_skills
    extra = resume_skills - job_skills

    match_pct = (len(matching) / len(job_skills) * 100) if job_skills else 0.0

    return {
        "matching_skills": sorted(matching),
        "missing_skills": sorted(missing),
        "extra_skills": sorted(extra),
        "match_percentage": round(match_pct, 1),
        "resume_skill_count": len(resume_skills),
        "job_skill_count": len(job_skills),
    }


if __name__ == "__main__":
    # Quick demo
    sample_resume = """
    Experienced software engineer with 3 years of Python and JavaScript development.
    Proficient in React, Django, and PostgreSQL. Familiar with Docker, AWS, and CI/CD
    pipelines. Strong communication and teamwork skills. BS in Computer Science.
    """

    sample_job = """
    Looking for a full-stack developer with expertise in Python, React, and Node.js.
    Must have experience with AWS, Docker, and Kubernetes. Knowledge of machine learning
    and data analysis is a plus. Strong problem solving and leadership skills required.
    """

    resume_skills = extract_skills(sample_resume)
    job_skills = extract_skills(sample_job)
    gap = analyze_skill_gap(resume_skills, job_skills)

    print(f"Resume skills: {resume_skills}")
    print(f"Job skills:    {job_skills}")
    print(f"\nSkill gap analysis:")
    print(f"  Matching:   {gap['matching_skills']}")
    print(f"  Missing:    {gap['missing_skills']}")
    print(f"  Extra:      {gap['extra_skills']}")
    print(f"  Match %:    {gap['match_percentage']}%")
