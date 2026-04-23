"""
evaluation/metrics.py — Coherence, diversity, and topic quality metrics
"""

import numpy as np
import pandas as pd
from gensim.models import CoherenceModel
import config


# ── Coherence ────────────────────────────────────────────────────────────────

def lda_coherence(lda_model, tokenized_docs, dictionary, metric="c_v") -> float:
    """
    Compute coherence score for a trained LDA model.

    Args:
        lda_model      : gensim LdaMulticore model
        tokenized_docs : list of token lists
        dictionary     : gensim Dictionary
        metric         : 'c_v' | 'u_mass' | 'c_uci' | 'c_npmi'
    Returns:
        float coherence score
    """
    cm    = CoherenceModel(
        model=lda_model,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence=metric,
    )
    score = cm.get_coherence()
    print(f"[Eval] LDA Coherence ({metric}): {score:.4f}")
    return score


def sweep_lda_coherence(cleaned_docs, topic_range=None) -> pd.DataFrame:
    """
    Train LDA models for different topic counts and record coherence.

    Returns:
        DataFrame with columns [num_topics, coherence]
    """
    from models.lda_model import LDATopicModel

    topic_range = topic_range or config.TUNE_TOPICS_RANGE
    tokenized   = [doc.split() for doc in cleaned_docs]
    records     = []

    for k in topic_range:
        print(f"[Sweep] num_topics={k}")
        m = LDATopicModel(num_topics=k, passes=10)
        m.fit(cleaned_docs)
        score = m.coherence_score(cleaned_docs)
        records.append({"num_topics": k, "coherence": score})

    return pd.DataFrame(records)


# ── Topic Diversity ───────────────────────────────────────────────────────────

def topic_diversity(topics_dict: dict, top_n: int = 10) -> float:
    """
    Fraction of unique words across top-N words of all topics.
    Higher = more diverse topics (less word overlap).

    Args:
        topics_dict : {topic_id: [word, …]}
        top_n       : how many top words per topic to consider
    Returns:
        float in [0, 1]
    """
    all_words    = []
    for words in topics_dict.values():
        all_words.extend(words[:top_n])

    if not all_words:
        return 0.0

    diversity = len(set(all_words)) / len(all_words)
    print(f"[Eval] Topic Diversity: {diversity:.4f}")
    return diversity


# ── Summary report ────────────────────────────────────────────────────────────

def evaluation_report(
    lda_model,
    cleaned_docs: list[str],
    lda_topics: dict,
    bert_topics: dict = None,
) -> dict:
    """
    Produce a consolidated evaluation report dict.
    """
    tokenized  = [doc.split() for doc in cleaned_docs]
    cv_score   = lda_coherence(lda_model.model, tokenized, lda_model.dictionary, "c_v")
    um_score   = lda_coherence(lda_model.model, tokenized, lda_model.dictionary, "u_mass")
    lda_div    = topic_diversity(lda_topics)
    perplexity = lda_model.perplexity()

    report = {
        "lda": {
            "num_topics":  lda_model.num_topics,
            "coherence_cv":    round(cv_score,   4),
            "coherence_umass": round(um_score,   4),
            "topic_diversity": round(lda_div,    4),
            "perplexity":      round(perplexity, 4),
        }
    }

    if bert_topics:
        bt_div = topic_diversity(bert_topics)
        report["bertopic"] = {
            "num_topics":      len(bert_topics),
            "topic_diversity": round(bt_div, 4),
        }

    return report
