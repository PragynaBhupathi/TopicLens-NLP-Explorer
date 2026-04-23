"""
config.py — Central configuration for Topic Modeling project
"""

# ─── Data ────────────────────────────────────────────────────────────────────
DATASET_SUBSET   = "train"   # 'train' | 'test' | 'all'
CATEGORIES       = None      # None = all 20; or list like ['sci.space', 'rec.autos']
REMOVE_META      = ('headers', 'footers', 'quotes')

# ─── Preprocessing ───────────────────────────────────────────────────────────
MIN_TOKEN_LENGTH = 3
MAX_DF           = 0.40      # drop tokens in >40% docs
MIN_DF           = 10        # drop tokens in <10 docs

# ─── LDA (Gensim LdaMulticore) ───────────────────────────────────────────────
LDA_NUM_TOPICS   = 20
LDA_PASSES       = 15
LDA_ALPHA        = "auto"
LDA_ETA          = "auto"
LDA_WORKERS      = 2
LDA_RANDOM_STATE = 42

# ─── BERTopic ────────────────────────────────────────────────────────────────
BERT_EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
BERT_NR_TOPICS         = "auto"
BERT_UMAP_NEIGHBORS    = 15
BERT_UMAP_COMPONENTS   = 5
BERT_HDBSCAN_MIN_SIZE  = 15
BERT_RANDOM_STATE      = 42

# ─── Evaluation ──────────────────────────────────────────────────────────────
COHERENCE_METRIC       = "c_v"          # 'c_v' | 'u_mass' | 'c_uci'
TOP_WORDS_PER_TOPIC    = 10

# ─── Tuning ──────────────────────────────────────────────────────────────────
TUNE_TOPICS_RANGE      = range(5, 35, 5) # 5, 10, 15, 20, 25, 30

# ─── Output ──────────────────────────────────────────────────────────────────
OUTPUT_DIR             = "outputs/"
MODEL_SAVE_DIR         = "outputs/saved_models/"
LDA_VIS_PATH           = "outputs/lda_vis.html"
COHERENCE_PLOT_PATH    = "outputs/coherence_scores.png"
TOPIC_WORDS_CSV        = "outputs/topic_words.csv"

# ─── API ─────────────────────────────────────────────────────────────────────
API_HOST               = "0.0.0.0"
API_PORT               = 5000
API_DEBUG              = False
