"""
visualization/plots.py — All plotting utilities (pyLDAvis, coherence curve, word clouds, topic bars)
"""

import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

import config


# ── pyLDAvis ─────────────────────────────────────────────────────────────────

def save_lda_vis(lda_model, corpus, dictionary, path=None):
    """Save interactive pyLDAvis HTML."""
    path = path or config.LDA_VIS_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    vis  = gensimvis.prepare(lda_model, corpus, dictionary, mds="mmds")
    pyLDAvis.save_html(vis, path)
    print(f"[Viz] LDA visualization saved → {path}")


# ── Coherence curve ──────────────────────────────────────────────────────────

def plot_coherence_curve(df: pd.DataFrame, path=None):
    """
    Plot coherence vs num_topics and save to file.

    Args:
        df   : DataFrame with columns [num_topics, coherence]
        path : output file path
    """
    path = path or config.COHERENCE_PLOT_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["num_topics"], df["coherence"], marker="o", linewidth=2,
            color="#4F8EF7", markersize=7)

    best_idx = df["coherence"].idxmax()
    best_k   = df.loc[best_idx, "num_topics"]
    best_cv  = df.loc[best_idx, "coherence"]
    ax.axvline(best_k, linestyle="--", color="#E05C5C", alpha=0.7,
               label=f"Best k={best_k} (Cv={best_cv:.3f})")

    ax.set_xlabel("Number of Topics", fontsize=12)
    ax.set_ylabel("Coherence Score (Cv)", fontsize=12)
    ax.set_title("LDA Coherence vs Number of Topics", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Viz] Coherence curve saved → {path}")


# ── Word Clouds ───────────────────────────────────────────────────────────────

def save_wordclouds(topics_dict: dict, out_dir: str = "outputs/wordclouds/"):
    """Generate one word-cloud image per topic."""
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for tid, words in topics_dict.items():
        wc = WordCloud(
            width=600, height=300,
            background_color="white",
            colormap="viridis",
            max_words=50,
        ).generate(" ".join(words))
        path = os.path.join(out_dir, f"topic_{tid}.png")
        wc.to_file(path)
        saved.append(path)
    print(f"[Viz] {len(saved)} word clouds saved → {out_dir}")
    return saved


# ── Topic word-score bar charts ───────────────────────────────────────────────

def plot_topic_words(lda_model_obj, num_topics: int = 6, num_words: int = 10,
                     path: str = "outputs/topic_words.png"):
    """Plot top words for the first N topics as horizontal bar charts."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    topics_raw = lda_model_obj.model.show_topics(
        num_topics=num_topics, num_words=num_words, formatted=False
    )

    cols = 3
    rows = (num_topics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5))
    axes = axes.flatten()

    colors = cm.tab10(np.linspace(0, 1, num_topics))

    for i, (tid, word_scores) in enumerate(topics_raw):
        words  = [w for w, _ in word_scores][::-1]
        scores = [s for _, s in word_scores][::-1]
        axes[i].barh(words, scores, color=colors[i], edgecolor="white")
        axes[i].set_title(f"Topic {tid}", fontweight="bold", fontsize=11)
        axes[i].tick_params(axis="y", labelsize=9)
        axes[i].set_xlabel("Weight", fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Top Words per Topic (LDA)", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Topic word chart saved → {path}")


# ── Export topic words CSV ────────────────────────────────────────────────────

def export_topic_words_csv(topics_dict: dict, path=None):
    path = path or config.TOPIC_WORDS_CSV
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = [
        {"topic_id": tid, "rank": rank + 1, "word": word}
        for tid, words in topics_dict.items()
        for rank, word in enumerate(words)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[Viz] Topic words CSV saved → {path}")
