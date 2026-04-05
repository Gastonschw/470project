# ResuRank AI

AI-powered resume-to-job ranking system built for CSCE 470 (Information Retrieval).

## Overview

ResuRank AI takes a resume and ranks ~30k real job postings by how well they match the candidate's skills and experience. It uses three ranking strategies and compares their results:

1. **TF-IDF + Cosine Similarity** — lexical baseline that matches based on word overlap
2. **Sentence-Transformer Embeddings** — semantic search using the all-MiniLM-L6-v2 model to understand meaning beyond exact word matches
3. **Hybrid Blend** — weighted combination of both TF-IDF and semantic scores for the best of both approaches

For each ranked job, the system also performs:
- **Skill Extraction** — pulls out skills from both the resume and job descriptions using a built-in taxonomy of ~100 technical and professional skills
- **Skill Gap Analysis** — shows which skills match, which are missing, and what percentage of the job's required skills the candidate has

## Dataset

Uses the CareerBuilder Job Listings 2020 dataset (~30k job postings) in LDJSON (line-delimited JSON) format. Each record contains fields like `job_title`, `job_description`, `company_name`, `city`, `state`, etc.

The data file should be located one directory up from this folder:
```
470proj/
  marketing_sample_for_careerbuilder_usa-careerbuilder_job_listing__20200401_20200630__30k_data.ldjson
  resurank-ai/
    test_core.py
    ...
```

## Setup

### 1. Install Python dependencies

Make sure you have Python 3.10+ installed, then run:

```bash
cd resurank-ai
pip install -r requirements.txt
```

This installs: pandas, scikit-learn, numpy, sentence-transformers (includes PyTorch), PyPDF2, and python-docx.

### 2. Verify the dataset is in place

The LDJSON data file should be in the parent directory (`470proj/`). The test script references it with a relative path.

## How to Run

### Option 1: Quick Demo (no resume file needed)

This uses a built-in sample resume hardcoded in `test_core.py`:

```bash
python test_core.py --data ../marketing_sample_for_careerbuilder_usa-careerbuilder_job_listing__20200401_20200630__30k_data.ldjson --demo
```

This will:
1. Load the sample resume and extract skills from it
2. Load and preprocess 500 job postings from the dataset (default sample size)
3. Run TF-IDF ranking and show the top 5 matches with skill gap analysis
4. Run Semantic ranking (downloads the model on first run, ~90MB) and show top 5
5. Run Hybrid ranking (combines both scores) and show top 5
6. Print a summary with timing info

### Option 2: Use Your Own Resume

Provide a path to your resume as a PDF or DOCX file:

```bash
python test_core.py --data ../marketing_sample_for_careerbuilder_usa-careerbuilder_job_listing__20200401_20200630__30k_data.ldjson --resume path/to/your_resume.pdf
```

Supported formats:
- `.pdf` — parsed using PyPDF2
- `.docx` — parsed using python-docx

The parser extracts all text from the file, then the system cleans it and runs it through the ranking pipeline.

### Optional Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--sample N` | 500 | Number of job postings to sample from the dataset. Use a smaller number (e.g. 100) for faster testing, or remove to use all ~30k. |
| `--top-k N` | 10 | Number of top-ranked results to return from each ranker. |
| `--alpha F` | 0.5 | Hybrid blend weight. 0.0 = pure semantic, 1.0 = pure TF-IDF, 0.5 = equal blend. |

### Example with all flags

```bash
python test_core.py --data ../marketing_sample_for_careerbuilder_usa-careerbuilder_job_listing__20200401_20200630__30k_data.ldjson --demo --sample 1000 --top-k 5 --alpha 0.7
```

This samples 1000 jobs, returns top 5 results, and weights TF-IDF at 70% in the hybrid score.

## Understanding the Output

For each ranker, the output shows the top matched jobs with:

```
  #1 [Score: 0.3008]
  Title: Software Developer
  Description: diversified services network, inc. (dsn) is seeking a software developer...
  Skill Match: 33.3% (2/6)
  Matching: java, python
  Missing:  aws, communication, dynamodb, excel
```

- **Score** — similarity score between the resume and job (higher = better match)
- **Title** — the job title from the dataset
- **Description** — first 100 characters of the cleaned job description
- **Skill Match** — percentage of the job's required skills found in the resume
- **Matching** — skills present in both the resume and job description
- **Missing** — skills the job asks for that aren't in the resume

The Hybrid ranker also shows the breakdown between TF-IDF and Semantic scores:
```
  #1 [Score: 1.0000] (TF-IDF: 0.301, Semantic: 0.395)
```

## Running Individual Modules

You can also run each module standalone:

```bash
# Test the data loader on its own
python data_loader.py ../marketing_sample_for_careerbuilder_usa-careerbuilder_job_listing__20200401_20200630__30k_data.ldjson

# Test skill extraction with built-in examples
python skill_extractor.py

# Test evaluation metrics with synthetic data
python evaluation.py

# Parse a resume file and print extracted text
python resume_parser.py path/to/resume.pdf
```

## Project Structure

| File | Description |
|------|-------------|
| `data_loader.py` | Loads the LDJSON dataset, cleans text (strips HTML, URLs, special chars), filters short descriptions |
| `resume_parser.py` | Extracts text from PDF (PyPDF2) and DOCX (python-docx) resume files |
| `skill_extractor.py` | Matches skills from a ~100-skill taxonomy against text, computes skill gap metrics |
| `ranker.py` | Three ranker classes: TFIDFRanker, SemanticRanker, HybridRanker |
| `evaluation.py` | Precision@k, Recall@k, Average Precision metrics + LLM-based ground truth generation helpers |
| `test_core.py` | Main entry point — ties everything together into an end-to-end demo |
| `requirements.txt` | Python package dependencies |

## Evaluation

To compute precision@k and other IR metrics, you need ground truth relevance labels. The system includes helpers for this:

1. Run `create_eval_dataset_template()` from `evaluation.py` to generate a JSON file with (resume, job) pairs and LLM prompts
2. Feed each prompt to an LLM (GPT/Claude) to get relevance judgments (`{"relevant": true/false}`)
3. Save the labeled results back into the JSON file
4. Load labels with `load_eval_labels()` and compute metrics with `evaluate_ranker()`

## Notes

- First run will download the sentence-transformers model (~90MB). Subsequent runs use the cached version.
- The dataset has 1 malformed line (line 6614 — two JSON objects concatenated without a newline). The data loader skips it automatically.
- Sampling fewer jobs (e.g. `--sample 100`) makes testing much faster since semantic embedding computation scales with dataset size.
