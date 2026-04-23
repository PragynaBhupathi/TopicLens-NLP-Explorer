"""
data/loader.py — Dataset loading utilities
"""

from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import config


def load_20newsgroups(subset=None, categories=None):
    """
    Load the 20 Newsgroups dataset.

    Returns:
        docs        : List[str]  — raw documents
        labels      : List[int]  — numeric class labels
        label_names : List[str]  — human-readable category names
    """
    subset     = subset     or config.DATASET_SUBSET
    categories = categories or config.CATEGORIES

    dataset = fetch_20newsgroups(
        subset=subset,
        categories=categories,
        remove=config.REMOVE_META,
        random_state=42,
    )

    docs        = dataset.data
    labels      = dataset.target.tolist()
    label_names = dataset.target_names

    # Basic cleanup: drop empty / very-short documents
    valid = [(d, l) for d, l in zip(docs, labels) if len(d.strip()) > 30]
    docs, labels = zip(*valid) if valid else ([], [])

    print(f"[Loader] Loaded {len(docs)} documents | {len(set(labels))} categories")
    return list(docs), list(labels), label_names


def load_custom(file_path: str, text_col: str = "text"):
    """
    Load a custom CSV dataset.

    Args:
        file_path : path to CSV file
        text_col  : column name that contains the document text
    Returns:
        docs : List[str]
    """
    df   = pd.read_csv(file_path)
    docs = df[text_col].dropna().tolist()
    print(f"[Loader] Loaded {len(docs)} documents from {file_path}")
    return docs
