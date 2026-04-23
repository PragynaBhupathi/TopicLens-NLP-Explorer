"""
main.py — Full pipeline entry point

Usage:
    python main.py                          # Train LDA + BERTopic, save, visualize
    python main.py --model lda             # LDA only
    python main.py --model bertopic        # BERTopic only
    python main.py --model lda --tune      # Coherence sweep to find best k
    python main.py --model lda --topics 15 # Override num_topics
"""

import argparse
import os
import sys

import config
from data.loader import load_20newsgroups
from preprocessing.text_cleaner import preprocess
from models.lda_model import LDATopicModel
from models.bertopic_model import BERTopicModel
from evaluation.metrics import evaluation_report, sweep_lda_coherence
from visualization.plots import (
    save_lda_vis,
    plot_coherence_curve,
    save_wordclouds,
    plot_topic_words,
    export_topic_words_csv,
)


def parse_args():
    p = argparse.ArgumentParser(description="Topic Modeling Pipeline")
    p.add_argument("--model",   choices=["lda", "bertopic", "both"], default="lda",
                   help="Which model to train (default: lda)")
    p.add_argument("--topics",  type=int, default=None,
                   help="Override num_topics (LDA)")
    p.add_argument("--tune",    action="store_true",
                   help="Sweep topic counts and plot coherence curve (LDA only)")
    p.add_argument("--no-save", action="store_true",
                   help="Skip saving model to disk")
    p.add_argument("--no-vis",  action="store_true",
                   help="Skip all visualizations")
    return p.parse_args()


def run_lda(cleaned_docs, raw_docs, args):
    print("\n" + "="*55)
    print(" LDA PIPELINE")
    print("="*55)

    num_topics = args.topics or config.LDA_NUM_TOPICS
    lda = LDATopicModel(num_topics=num_topics)
    lda.fit(cleaned_docs)

    topics_dict = lda.get_topics_dict()

    # Evaluation
    print("\n[Eval] Running evaluation …")
    tokenized = [doc.split() for doc in cleaned_docs]
    from gensim.models import CoherenceModel
    cm = CoherenceModel(
        model=lda.model,
        texts=tokenized,
        dictionary=lda.dictionary,
        coherence="c_v",
    )
    cv = cm.get_coherence()
    print(f"  Coherence (Cv)  : {cv:.4f}")
    print(f"  Perplexity      : {lda.perplexity():.4f}")

    # Save
    if not args.no_save:
        lda.save()

    # Visualizations
    if not args.no_vis:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        save_lda_vis(lda.model, lda.corpus, lda.dictionary, config.LDA_VIS_PATH)
        plot_topic_words(lda, num_topics=min(num_topics, 9))
        save_wordclouds(topics_dict)
        export_topic_words_csv(topics_dict)

    # Print top topics
    print("\n[Topics] Top words per topic:")
    for tid, words in list(topics_dict.items())[:6]:
        print(f"  Topic {tid:>2}: {' | '.join(words[:8])}")

    return lda, topics_dict


def run_tune(cleaned_docs):
    print("\n[Tune] Sweeping topic counts …")
    df = sweep_lda_coherence(cleaned_docs)
    print(df.to_string(index=False))
    plot_coherence_curve(df, config.COHERENCE_PLOT_PATH)
    best = df.loc[df["coherence"].idxmax()]
    print(f"\n[Tune] Best k = {int(best['num_topics'])} (Cv = {best['coherence']:.4f})")
    return df


def run_bertopic(raw_docs, args):
    print("\n" + "="*55)
    print(" BERTOPIC PIPELINE")
    print("="*55)

    bt = BERTopicModel()
    bt.fit(raw_docs)

    topics_dict = bt.get_topics_dict()
    topic_info  = bt.get_topic_info()
    print(f"\n[BERTopic] Topic info:\n{topic_info.head(10).to_string(index=False)}")

    if not args.no_save:
        bt.save()

    return bt, topics_dict


def main():
    args = parse_args()

    os.makedirs(config.OUTPUT_DIR,    exist_ok=True)
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[Pipeline] Loading dataset …")
    raw_docs, labels, label_names = load_20newsgroups()

    # ── Preprocess ────────────────────────────────────────────────────────────
    print("[Pipeline] Preprocessing …")
    cleaned_docs = preprocess(raw_docs)

    # ── Tune (optional) ───────────────────────────────────────────────────────
    if args.tune:
        run_tune(cleaned_docs)
        sys.exit(0)

    # ── Train ─────────────────────────────────────────────────────────────────
    lda_model = bt_model = None
    lda_topics = bt_topics = None

    if args.model in ("lda", "both"):
        lda_model, lda_topics = run_lda(cleaned_docs, raw_docs, args)

    if args.model in ("bertopic", "both"):
        bt_model, bt_topics = run_bertopic(raw_docs, args)

    # ── Final report ──────────────────────────────────────────────────────────
    if lda_model:
        print("\n" + "="*55)
        print(" EVALUATION REPORT")
        print("="*55)
        report = evaluation_report(lda_model, cleaned_docs, lda_topics, bt_topics)
        import json
        print(json.dumps(report, indent=2))

    print("\n✅ Pipeline complete!")
    print(f"   Outputs → {os.path.abspath(config.OUTPUT_DIR)}")
    print(f"   Saved models → {os.path.abspath(config.MODEL_SAVE_DIR)}")
    print(f"\n   To launch the dashboard:")
    print(f"     python api/app.py")
    print(f"     Then open: http://localhost:{config.API_PORT}")


if __name__ == "__main__":
    main()
