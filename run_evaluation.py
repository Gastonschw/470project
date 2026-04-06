"""
Run all 3 rankers against all 3 resumes and print a comparison table.
Only loads/fits on the job indices present in the eval datasets.
Usage: python run_evaluation.py
"""

import os
import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", message=".*unauthenticated.*HF Hub.*")

from data_loader import load_job_data
from resume_parser import parse_resume
from ranker import TFIDFRanker, SemanticRanker, HybridRanker
from evaluation import load_eval_labels, evaluate_ranker

# ── Paths ────────────────────────────────────────────────────────────────
DATA_PATH = "../marketing_sample_for_careerbuilder_usa-careerbuilder_job_listing__20200401_20200630__30k_data.ldjson"

RESUMES = {
    "cody":      "../Cody_Fredrickson_Resume (2).docx",
    "malik":     "../Malik_Rabb_Resume (1).docx",
    "sriharsha": "../Jandhyala_Sriharsha_1b.pdf",
}

EVAL_FILES = {
    "cody":      "eval_cody.json",
    "malik":     "eval_malik.json",
    "sriharsha": "eval_sriharsha.json",
}

K_VALUES = [5, 10]

# ── Step 1: Load eval labels & collect all needed job indices ────────────
print("Loading eval labels...")
eval_labels = {}
relevant_sets = {}
all_eval_indices = set()

for name, path in EVAL_FILES.items():
    labels = load_eval_labels(path)
    eval_labels[name] = labels
    relevant_sets[name] = {idx for idx, rel in labels.items() if rel}
    all_eval_indices.update(labels.keys())

print(f"  {len(all_eval_indices)} unique job indices across all eval sets")

# ── Step 2: Load job data, subset to eval indices only ───────────────────
print("Loading job data...")
df = load_job_data(DATA_PATH)

eval_indices_sorted = sorted(all_eval_indices)
df_subset = df.loc[eval_indices_sorted].copy()
descriptions = df_subset["job_description_clean"].tolist()

# Mapping: position in subset list -> original job index
pos_to_orig = {pos: orig_idx for pos, orig_idx in enumerate(eval_indices_sorted)}

print(f"  Subset: {len(descriptions)} job descriptions for evaluation")

# ── Step 3: Parse resumes ────────────────────────────────────────────────
print("Parsing resumes...")
resume_texts = {name: parse_resume(path) for name, path in RESUMES.items()}

# ── Step 4: Fit rankers on subset (once) ─────────────────────────────────
print("Fitting rankers...")
tfidf = TFIDFRanker()
tfidf.fit(descriptions)

semantic = SemanticRanker()
semantic.fit(descriptions)

hybrid = HybridRanker()
hybrid.fit(descriptions)

rankers = {"TF-IDF": tfidf, "Semantic": semantic, "Hybrid": hybrid}

# ── Step 5: Rank & evaluate ──────────────────────────────────────────────
print("Ranking and evaluating...\n")
results = []

for resume_name, resume_text in resume_texts.items():
    relevant = relevant_sets[resume_name]
    for ranker_name, ranker in rankers.items():
        ranked = ranker.rank(resume_text, top_k=len(descriptions))

        # HybridRanker returns list[dict]; convert to list[tuple]
        if isinstance(ranked[0], dict):
            ranked = [(d["job_index"], d["final_score"]) for d in ranked]

        # Remap subset positions back to original job indices
        ranked = [(pos_to_orig[pos], score) for pos, score in ranked]

        metrics = evaluate_ranker(ranked, relevant, k_values=K_VALUES)
        results.append((resume_name, ranker_name, metrics))

# ── Step 6: Print comparison table ───────────────────────────────────────
header = f"{'Resume':<12} {'Ranker':<10} {'P@5':>6} {'P@10':>6} {'R@5':>6} {'R@10':>6} {'AP':>6}"
print(header)
print("-" * len(header))

for resume_name, ranker_name, m in results:
    print(
        f"{resume_name:<12} {ranker_name:<10} "
        f"{m['P@5']:>6.3f} {m['P@10']:>6.3f} "
        f"{m['R@5']:>6.3f} {m['R@10']:>6.3f} "
        f"{m['AP']:>6.3f}"
    )
