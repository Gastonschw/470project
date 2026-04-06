"""
data_loader.py
Loads and preprocesses the CareerBuilder job postings dataset.
"""

import re
import pandas as pd


def load_job_data(data_path: str, sample_size: int = None) -> pd.DataFrame:
    """
    Load job postings from the CareerBuilder LDJSON dataset.

    Args:
        data_path: Path to the LDJSON data file.
        sample_size: Optional number of rows to sample (for faster testing).

    Returns:
        DataFrame with cleaned job postings.
    """
    # Read LDJSON line-by-line, skipping malformed lines (e.g. line 6614
    # has two JSON objects concatenated without a newline separator)
    import json
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(records)

    # Standardize column names to lowercase
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Try to identify the description column (dataset may use different names)
    desc_col = None
    for candidate in ["job_description", "description", "jobdescription", "job_desc"]:
        if candidate in df.columns:
            desc_col = candidate
            break

    if desc_col is None:
        print(f"Available columns: {list(df.columns)}")
        raise ValueError("Could not find a job description column in the dataset.")

    # Rename to standard name if needed
    if desc_col != "job_description":
        df = df.rename(columns={desc_col: "job_description"})

    # Drop rows with missing descriptions
    df = df.dropna(subset=["job_description"])

    # Clean description text
    df["job_description_clean"] = df["job_description"].apply(clean_text)

    # Drop rows where cleaned description is too short (likely garbage)
    df = df[df["job_description_clean"].str.split().str.len() >= 20]

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)

    df = df.reset_index(drop=True)
    return df


def clean_text(text: str) -> str:
    """
    Clean and normalize text for IR processing.

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s.,;:()\-/+#]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <path_to_data>")
        sys.exit(1)
    df = load_job_data(sys.argv[1], sample_size=100)
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample description (first 200 chars):")
    print(df["job_description_clean"].iloc[0][:200])
