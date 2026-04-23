"""
models/lda_model.py — Gensim LDA wrapper with corpus building, fit, inference, save/load
"""

import os
import pickle
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import config


class LDATopicModel:
    """
    Wraps Gensim LdaMulticore.

    Usage:
        model = LDATopicModel(num_topics=20)
        model.fit(cleaned_docs)
        print(model.get_topics())
        print(model.coherence_score(cleaned_docs))
    """

    def __init__(
        self,
        num_topics:   int = None,
        passes:       int = None,
        alpha:        str = None,
        eta:          str = None,
        workers:      int = None,
        random_state: int = None,
    ):
        self.num_topics   = num_topics   or config.LDA_NUM_TOPICS
        self.passes       = passes       or config.LDA_PASSES
        self.alpha        = alpha        or config.LDA_ALPHA
        self.eta          = eta          or config.LDA_ETA
        self.workers      = workers      or config.LDA_WORKERS
        self.random_state = random_state or config.LDA_RANDOM_STATE

        self.model      = None
        self.corpus     = None
        self.dictionary = None

    # ── Corpus construction ─────────────────────────────────────────────────

    def _build_corpus(self, cleaned_docs: list[str]):
        tokenized = [doc.split() for doc in cleaned_docs]
        self.dictionary = corpora.Dictionary(tokenized)
        self.dictionary.filter_extremes(
            no_below=config.MIN_DF,
            no_above=config.MAX_DF,
        )
        self.corpus = [self.dictionary.doc2bow(t) for t in tokenized]
        return tokenized

    # ── Fit ─────────────────────────────────────────────────────────────────

    def fit(self, cleaned_docs: list[str]) -> "LDATopicModel":
        print(f"[LDA] Building corpus for {len(cleaned_docs)} docs …")
        self._build_corpus(cleaned_docs)
        print(f"[LDA] Vocabulary size: {len(self.dictionary)}")
        print(f"[LDA] Training with {self.num_topics} topics, {self.passes} passes …")

        # Use LdaModel for auto-tuning (alpha/eta="auto"), LdaMulticore otherwise
        if self.alpha == "auto" or self.eta == "auto":
            print("[LDA] Using LdaModel for auto-tuning alpha/eta …")
            self.model = gensim.models.LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=self.num_topics,
                passes=self.passes,
                alpha=self.alpha,
                eta=self.eta,
                random_state=self.random_state,
            )
        else:
            self.model = gensim.models.LdaMulticore(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=self.num_topics,
                passes=self.passes,
                alpha=self.alpha,
                eta=self.eta,
                workers=self.workers,
                random_state=self.random_state,
            )
        print("[LDA] Training complete.")
        return self

    # ── Inference ───────────────────────────────────────────────────────────

    def get_topics(self, num_words: int = None) -> list[tuple]:
        """Return list of (topic_id, word_list) tuples."""
        num_words = num_words or config.TOP_WORDS_PER_TOPIC
        raw = self.model.show_topics(
            num_topics=self.num_topics,
            num_words=num_words,
            formatted=False,
        )
        return [(tid, [w for w, _ in words]) for tid, words in raw]

    def get_topics_dict(self, num_words: int = None) -> dict:
        """Return {topic_id: [word, word, …]} dict."""
        return {tid: words for tid, words in self.get_topics(num_words)}

    def infer(self, cleaned_doc: str) -> list[tuple]:
        """Infer topic distribution for a single new document."""
        bow = self.dictionary.doc2bow(cleaned_doc.split())
        return self.model.get_document_topics(bow)

    # ── Evaluation ──────────────────────────────────────────────────────────

    def coherence_score(
        self,
        cleaned_docs: list[str],
        metric: str = None,
    ) -> float:
        metric    = metric or config.COHERENCE_METRIC
        tokenized = [doc.split() for doc in cleaned_docs]
        cm = CoherenceModel(
            model=self.model,
            texts=tokenized,
            dictionary=self.dictionary,
            coherence=metric,
        )
        score = cm.get_coherence()
        print(f"[LDA] Coherence ({metric}): {score:.4f}")
        return score

    def perplexity(self) -> float:
        """Log-perplexity on the training corpus (lower is better)."""
        return self.model.log_perplexity(self.corpus)

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, directory: str = None):
        directory = directory or config.MODEL_SAVE_DIR
        os.makedirs(directory, exist_ok=True)
        self.model.save(os.path.join(directory, "lda.model"))
        self.dictionary.save(os.path.join(directory, "lda.dict"))
        with open(os.path.join(directory, "lda_corpus.pkl"), "wb") as f:
            pickle.dump(self.corpus, f)
        print(f"[LDA] Model saved to {directory}")

    @classmethod
    def load(cls, directory: str = None) -> "LDATopicModel":
        directory = directory or config.MODEL_SAVE_DIR
        obj            = cls.__new__(cls)
        obj.model      = gensim.models.LdaMulticore.load(
            os.path.join(directory, "lda.model")
        )
        obj.dictionary = corpora.Dictionary.load(
            os.path.join(directory, "lda.dict")
        )
        with open(os.path.join(directory, "lda_corpus.pkl"), "rb") as f:
            obj.corpus = pickle.load(f)
        obj.num_topics = obj.model.num_topics
        print(f"[LDA] Model loaded from {directory}")
        return obj
