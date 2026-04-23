"""
models/bertopic_model.py — BERTopic wrapper with fit, inference, save/load
"""

import os
import config

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer


class BERTopicModel:
    """
    Wraps BERTopic with explicit sub-model configuration.

    Usage:
        model = BERTopicModel(nr_topics=20)
        model.fit(raw_docs)
        print(model.get_topic_info())
    """

    def __init__(
        self,
        embedding_model: str = None,
        nr_topics=None,
        umap_neighbors: int = None,
        umap_components: int = None,
        hdbscan_min_size: int = None,
        random_state: int = None,
    ):
        self.embedding_model_name = embedding_model  or config.BERT_EMBEDDING_MODEL
        self.nr_topics            = nr_topics        or config.BERT_NR_TOPICS
        self.umap_neighbors       = umap_neighbors   or config.BERT_UMAP_NEIGHBORS
        self.umap_components      = umap_components  or config.BERT_UMAP_COMPONENTS
        self.hdbscan_min_size     = hdbscan_min_size or config.BERT_HDBSCAN_MIN_SIZE
        self.random_state         = random_state     or config.BERT_RANDOM_STATE

        self.model  = None
        self.topics = None
        self.probs  = None

    def _build_model(self) -> BERTopic:
        embedding_model = SentenceTransformer(self.embedding_model_name)

        umap_model = UMAP(
            n_neighbors=self.umap_neighbors,
            n_components=self.umap_components,
            min_dist=0.0,
            metric="cosine",
            random_state=self.random_state,
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=self.hdbscan_min_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

        vectorizer_model = CountVectorizer(
            stop_words="english",
            min_df=config.MIN_DF,
            max_df=config.MAX_DF,
            ngram_range=(1, 2),
        )

        return BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            nr_topics=self.nr_topics,
            top_n_words=config.TOP_WORDS_PER_TOPIC,
            calculate_probabilities=True,
            verbose=True,
        )

    # ── Fit ─────────────────────────────────────────────────────────────────

    def fit(self, docs: list[str]) -> "BERTopicModel":
        """Fit BERTopic on raw (un-cleaned) documents."""
        print(f"[BERTopic] Building model …")
        self.model = self._build_model()
        print(f"[BERTopic] Fitting on {len(docs)} documents …")
        self.topics, self.probs = self.model.fit_transform(docs)
        n = len(set(t for t in self.topics if t != -1))
        print(f"[BERTopic] Found {n} topics (excluding outlier topic -1).")
        return self

    # ── Inference ───────────────────────────────────────────────────────────

    def get_topic_info(self):
        """DataFrame: topic id, count, name, top words."""
        return self.model.get_topic_info()

    def get_topics_dict(self, num_words: int = None) -> dict:
        """Return {topic_id: [word, …]} dict."""
        num_words = num_words or config.TOP_WORDS_PER_TOPIC
        result = {}
        for tid in self.model.get_topics():
            if tid == -1:
                continue
            result[tid] = [w for w, _ in self.model.get_topic(tid)[:num_words]]
        return result

    def infer(self, docs: list[str]):
        """Predict topics for new documents."""
        topics, probs = self.model.transform(docs)
        return topics, probs

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, directory: str = None):
        directory = directory or config.MODEL_SAVE_DIR
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "bertopic_model")
        self.model.save(path, serialization="safetensors", save_ctfidf=True)
        print(f"[BERTopic] Model saved to {path}")

    @classmethod
    def load(cls, directory: str = None) -> "BERTopicModel":
        directory = directory or config.MODEL_SAVE_DIR
        path      = os.path.join(directory, "bertopic_model")
        obj       = cls.__new__(cls)
        obj.model = BERTopic.load(path)
        print(f"[BERTopic] Model loaded from {path}")
        return obj
