"""
evaluation.py
Evaluation utilities for ResuRank AI.

Computes precision@k, recall@k, and NDCG using ground truth labels.
Includes a helper to generate ground truth labels using an LLM (GPT/Claude).
"""

import json
import os

# ============================================================
# Evaluation Metrics
# ============================================================

def precision_at_k(ranked_indices: list[int], relevant_set: set[int], k: int) -> float:
    """
    Compute Precision@K.

    Args:
        ranked_indices: Ordered list of job indices returned by the ranker.
        relevant_set: Set of job indices that are truly relevant.
        k: Number of top results to consider.

    Returns:
        Precision@K as a float in [0, 1].
    """
    top_k = ranked_indices[:k]
    relevant_in_top_k = sum(1 for idx in top_k if idx in relevant_set)
    return relevant_in_top_k / k


def recall_at_k(ranked_indices: list[int], relevant_set: set[int], k: int) -> float:
    """
    Compute Recall@K.

    Args:
        ranked_indices: Ordered list of job indices returned by the ranker.
        relevant_set: Set of job indices that are truly relevant.
        k: Number of top results to consider.

    Returns:
        Recall@K as a float in [0, 1].
    """
    if not relevant_set:
        return 0.0
    top_k = ranked_indices[:k]
    relevant_in_top_k = sum(1 for idx in top_k if idx in relevant_set)
    return relevant_in_top_k / len(relevant_set)


def average_precision(ranked_indices: list[int], relevant_set: set[int]) -> float:
    """
    Compute Average Precision (AP) for a single query.

    Args:
        ranked_indices: Ordered list of job indices returned by the ranker.
        relevant_set: Set of job indices that are truly relevant.

    Returns:
        AP score.
    """
    if not relevant_set:
        return 0.0

    hits = 0
    sum_precision = 0.0
    for i, idx in enumerate(ranked_indices):
        if idx in relevant_set:
            hits += 1
            sum_precision += hits / (i + 1)

    return sum_precision / len(relevant_set)


def evaluate_ranker(ranked_results: list[tuple[int, float]],
                    relevant_set: set[int],
                    k_values: list[int] = None) -> dict:
    """
    Run full evaluation suite on ranked results.

    Args:
        ranked_results: List of (job_index, score) tuples from ranker.
        relevant_set: Set of relevant job indices (ground truth).
        k_values: List of K values to evaluate at.

    Returns:
        Dict of evaluation metrics.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    ranked_indices = [idx for idx, _ in ranked_results]

    metrics = {}
    for k in k_values:
        metrics[f"P@{k}"] = precision_at_k(ranked_indices, relevant_set, k)
        metrics[f"R@{k}"] = recall_at_k(ranked_indices, relevant_set, k)
    metrics["AP"] = average_precision(ranked_indices, relevant_set)

    return metrics


# ============================================================
# Ground Truth Generation via LLM
# ============================================================

def generate_eval_prompt(resume_text: str, job_description: str, job_title: str = "") -> str:
    """
    Create a prompt to ask an LLM whether a job is relevant to a resume.

    Args:
        resume_text: The candidate's resume text.
        job_description: The job posting description.
        job_title: Optional job title.

    Returns:
        Formatted prompt string.
    """
    prompt = f"""You are evaluating whether a job posting is relevant to a candidate's resume.
A job is "relevant" if the candidate's skills, experience, and background make them a
reasonable fit for the position (they could realistically apply and be considered).

RESUME:
{resume_text[:2000]}

JOB POSTING:
Title: {job_title}
Description: {job_description[:2000]}

Is this job relevant to this candidate? Respond with ONLY a JSON object:
{{"relevant": true/false, "confidence": "high"/"medium"/"low", "reason": "brief explanation"}}
"""
    return prompt


def create_eval_dataset_template(resume_text: str, job_postings: list[dict],
                                  sample_size: int = 50,
                                  output_path: str = "eval_dataset.json"):
    """
    Create a template eval dataset with (resume, job) pairs for LLM labeling.
    You can feed the generated prompts to GPT/Claude to get relevance labels.

    Args:
        resume_text: Resume text.
        job_postings: List of dicts with 'index', 'title', 'description' keys.
        sample_size: Number of job postings to sample.
        output_path: Path to save the eval template.
    """
    import random
    random.seed(42)

    if len(job_postings) > sample_size:
        sampled = random.sample(job_postings, sample_size)
    else:
        sampled = job_postings

    eval_pairs = []
    for job in sampled:
        eval_pairs.append({
            "job_index": job["index"],
            "job_title": job.get("title", ""),
            "prompt": generate_eval_prompt(
                resume_text,
                job["description"],
                job.get("title", "")
            ),
            "label": None,  # Fill this in with LLM response or manual label
        })

    with open(output_path, "w") as f:
        json.dump(eval_pairs, f, indent=2)

    print(f"Eval template saved to {output_path} with {len(eval_pairs)} pairs.")
    print("Next step: Run each 'prompt' through GPT/Claude and fill in the 'label' field.")


def load_eval_labels(eval_path: str) -> dict[int, bool]:
    """
    Load labeled eval dataset and return a mapping of job_index -> relevant (bool).

    Args:
        eval_path: Path to labeled eval JSON file.

    Returns:
        Dict mapping job indices to relevance labels.
    """
    with open(eval_path, "r") as f:
        eval_data = json.load(f)

    labels = {}
    for entry in eval_data:
        if entry.get("label") is not None:
            if isinstance(entry["label"], dict):
                labels[entry["job_index"]] = entry["label"].get("relevant", False)
            elif isinstance(entry["label"], bool):
                labels[entry["job_index"]] = entry["label"]

    return labels


if __name__ == "__main__":
    # Demo with synthetic data
    print("=== Evaluation Metrics Demo ===\n")

    # Simulate: ranker returned these 10 jobs in order
    ranked = [(5, 0.9), (12, 0.85), (3, 0.8), (7, 0.75), (1, 0.7),
              (9, 0.65), (15, 0.6), (2, 0.55), (8, 0.5), (11, 0.45)]

    # Ground truth: jobs 5, 3, 1, 8 are actually relevant
    relevant = {5, 3, 1, 8}

    metrics = evaluate_ranker(ranked, relevant)
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
