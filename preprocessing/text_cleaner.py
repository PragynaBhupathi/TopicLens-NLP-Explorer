"""
preprocessing/text_cleaner.py — Text cleaning & tokenization pipeline
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import config

# Download required NLTK data (idempotent)
for pkg in ("stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

_lemmatizer  = WordNetLemmatizer()
_stop_words  = set(stopwords.words("english"))

# Extra domain-specific noise words common in newsgroup posts
_EXTRA_STOPS = {
    "would", "could", "should", "said", "also", "one", "two", "like",
    "use", "get", "make", "know", "think", "way", "thing", "come",
    "much", "many", "well", "even", "still", "first", "last", "time",
}
_stop_words.update(_EXTRA_STOPS)


def _clean_single(text: str) -> str:
    """Clean and lemmatize a single document. Returns a space-joined token string."""
    # Lowercase
    text = text.lower()
    # Strip URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Strip email addresses
    text = re.sub(r"\S+@\S+", " ", text)
    # Keep only alphabetic characters
    text = re.sub(r"[^a-z\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [
        _lemmatizer.lemmatize(tok)
        for tok in text.split()
        if tok not in _stop_words and len(tok) >= config.MIN_TOKEN_LENGTH
    ]
    return " ".join(tokens)


def preprocess(docs: list[str], show_progress: bool = True) -> list[str]:
    """
    Preprocess a list of raw text documents.

    Args:
        docs          : raw text documents
        show_progress : show tqdm progress bar
    Returns:
        cleaned_docs  : list of cleaned, space-joined token strings
    """
    iterator = tqdm(docs, desc="Cleaning", unit="doc") if show_progress else docs
    cleaned  = [_clean_single(doc) for doc in iterator]

    # Remove empty docs produced by cleaning
    cleaned = [c if c.strip() else "empty document" for c in cleaned]
    print(f"[Preprocessor] Cleaned {len(cleaned)} documents")
    return cleaned


def tokenize(cleaned_docs: list[str]) -> list[list[str]]:
    """Convert cleaned doc strings to token lists (for Gensim)."""
    return [doc.split() for doc in cleaned_docs]
