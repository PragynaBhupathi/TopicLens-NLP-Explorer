"""
api/app.py — Flask REST API for Topic Modeling inference
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import config
from preprocessing.text_cleaner import preprocess
from models.lda_model import LDATopicModel

app = Flask(__name__, static_folder="../frontend/static")
CORS(app)

# ── Load model at startup ─────────────────────────────────────────────────────
_lda = None

def get_lda():
    global _lda
    if _lda is None:
        try:
            _lda = LDATopicModel.load(config.MODEL_SAVE_DIR)
        except Exception as e:
            print(f"[API] Warning: Could not load saved LDA model: {e}")
    return _lda


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("../frontend", "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": get_lda() is not None})


@app.route("/api/topics", methods=["GET"])
def get_all_topics():
    """Return all discovered topics and their top words."""
    lda = get_lda()
    if lda is None:
        return jsonify({"error": "Model not trained yet. Run main.py first."}), 503

    topics = lda.get_topics_dict(num_words=15)
    result = [
        {"topic_id": int(tid), "words": words}
        for tid, words in topics.items()
    ]
    return jsonify({"topics": result, "num_topics": len(result)})


@app.route("/api/infer", methods=["POST"])
def infer():
    """
    Infer topic distribution for a given text.

    Body: { "text": "your document text here" }
    Returns: { "topics": [{"topic_id": 0, "probability": 0.8}, ...] }
    """
    lda = get_lda()
    if lda is None:
        return jsonify({"error": "Model not trained yet. Run main.py first."}), 503

    body = request.get_json(force=True)
    text = body.get("text", "").strip()
    if not text:
        return jsonify({"error": "Field 'text' is required."}), 400

    cleaned     = preprocess([text], show_progress=False)[0]
    topic_dist  = lda.infer(cleaned)
    sorted_dist = sorted(topic_dist, key=lambda x: x[1], reverse=True)

    topics_dict = lda.get_topics_dict(num_words=10)
    result      = [
        {
            "topic_id":    int(tid),
            "probability": round(float(prob), 4),
            "words":       topics_dict.get(tid, []),
        }
        for tid, prob in sorted_dist
    ]
    return jsonify({"input_text": text[:200], "topics": result})


@app.route("/api/topic/<int:topic_id>", methods=["GET"])
def get_topic(topic_id: int):
    """Return detailed info about a single topic."""
    lda = get_lda()
    if lda is None:
        return jsonify({"error": "Model not loaded."}), 503

    topics = lda.get_topics_dict(num_words=20)
    if topic_id not in topics:
        return jsonify({"error": f"Topic {topic_id} not found."}), 404

    return jsonify({"topic_id": topic_id, "words": topics[topic_id]})


@app.route("/api/stats", methods=["GET"])
def stats():
    """Return model statistics."""
    lda = get_lda()
    if lda is None:
        return jsonify({"error": "Model not loaded."}), 503

    return jsonify({
        "num_topics":    lda.num_topics,
        "vocab_size":    len(lda.dictionary),
        "corpus_size":   len(lda.corpus),
        "passes":        lda.passes,
        "alpha":         str(lda.alpha),
    })


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=config.API_DEBUG,
    )
