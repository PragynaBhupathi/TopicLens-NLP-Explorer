# 🔍 TopicLens — NLP Topic Modeling Suite

> End-to-end, deployable NLP project for **unsupervised topic discovery** using LDA and BERTopic, with a REST API and an interactive web dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?style=flat-square)
![BERTopic](https://img.shields.io/badge/BERTopic-0.16-purple?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Problem Statement](#-problem-statement)
3. [Dataset](#-dataset)
4. [Project Architecture](#-project-architecture)
5. [Tech Stack](#-tech-stack)
6. [Project Structure](#-project-structure)
7. [Installation & Setup](#-installation--setup)
8. [Pipeline Walkthrough](#-pipeline-walkthrough)
9. [Models Used](#-models-used)
10. [Evaluation Metrics](#-evaluation-metrics)
11. [REST API Reference](#-rest-api-reference)
12. [Web Dashboard](#-web-dashboard)
13. [Docker Deployment](#-docker-deployment)
14. [How to Run](#-how-to-run)
15. [Expected Results](#-expected-results)
16. [Visualizations](#-visualizations)
17. [Configuration](#-configuration)
18. [Future Improvements](#-future-improvements)
19. [References](#-references)

---

## 📖 Project Overview

**TopicLens** is a production-ready NLP project that automatically discovers hidden themes inside large document collections — without any labels. It ships with:

- ✅ Full preprocessing pipeline (cleaning, lemmatization, stopword removal)
- ✅ Two models: classical **LDA** (Gensim) and modern **BERTopic** (transformers)
- ✅ Automated **evaluation** (coherence score, topic diversity, perplexity)
- ✅ Beautiful **visualizations** (pyLDAvis, word clouds, bar charts)
- ✅ **Flask REST API** with live inference endpoint
- ✅ Dark-themed interactive **web dashboard**
- ✅ **Docker** containerisation for one-command deployment

---

## 🎯 Problem Statement

**Task:** Given a raw, unlabeled collection of text documents, automatically identify the underlying topics discussed across the corpus.

**Why it matters:**
- News aggregators cluster articles by theme
- Search engines use topic signals to improve retrieval ranking
- Businesses use it for customer-feedback triage and trend detection
- Researchers use it to summarise large literature corpora

**Core challenges:**
- Fully unsupervised — no ground truth labels
- Noisy, real-world text (abbreviations, jargon, HTML artefacts)
- Selecting the optimal number of topics (hyperparameter tuning)
- Evaluation requires proxy metrics (coherence, diversity) instead of accuracy

---

## 📂 Dataset

### 20 Newsgroups
| Property | Value |
|----------|-------|
| Source | `sklearn.datasets.fetch_20newsgroups` |
| Size | ~18,000 newsgroup posts |
| Categories | 20 (sci, rec, comp, talk, misc, alt, soc) |
| Language | English |
| Format | Plain text |
| Metadata removed | headers, footers, quoted replies |

### All 20 Categories
```
comp.graphics              rec.autos                 sci.crypt
comp.os.ms-windows.misc    rec.motorcycles           sci.electronics
comp.sys.ibm.pc.hardware   rec.sport.baseball        sci.med
comp.sys.mac.hardware      rec.sport.hockey          sci.space
comp.windows.x             talk.politics.guns        talk.religion.misc
misc.forsale               talk.politics.mideast     alt.atheism
talk.politics.misc         soc.religion.christian
```

### Sample Document
```
From: user@university.edu
Subject: Need help with graphics card drivers

I recently installed a new graphics card and the drivers crash on
Windows 3.1. Has anyone encountered a similar issue? The BSOD
appears whenever I try to run a 3D application.
```

---

## 🏗️ Project Architecture

```
                      ┌─────────────────────────────┐
                      │     20 Newsgroups Dataset    │
                      └──────────────┬──────────────┘
                                     │ raw text
                                     ▼
                      ┌─────────────────────────────┐
                      │     Preprocessing Pipeline   │
                      │  lowercase → strip URLs      │
                      │  remove stopwords            │
                      │  lemmatize → filter tokens   │
                      └──────────┬──────────┬────────┘
                                 │          │
                    cleaned docs │          │ raw docs (BERTopic uses raw)
                                 ▼          ▼
               ┌────────────────────┐  ┌────────────────────┐
               │   LDA (Gensim)     │  │  BERTopic          │
               │                    │  │                    │
               │ Dictionary → BoW   │  │ SentenceTransformer│
               │ Gensim corpus      │  │ (all-MiniLM-L6-v2) │
               │ LdaMulticore       │  │ ↓                  │
               │ α, η auto-tuned    │  │ UMAP (768 → 5 dim) │
               └────────┬───────────┘  │ ↓                  │
                        │              │ HDBSCAN clustering  │
                        │              │ ↓                   │
                        │              │ c-TF-IDF topics     │
                        │              └────────┬────────────┘
                        │                       │
                        └──────────┬────────────┘
                                   ▼
                      ┌─────────────────────────────┐
                      │        Evaluation            │
                      │  Coherence (Cv, UMass)      │
                      │  Topic Diversity             │
                      │  Perplexity (LDA)            │
                      └──────────────┬──────────────┘
                                     │
                    ┌────────────────┼───────────────┐
                    ▼                ▼               ▼
             pyLDAvis HTML    Word Clouds      Topic CSV
                    │                               │
                    └──────────────┬────────────────┘
                                   ▼
                      ┌─────────────────────────────┐
                      │     Flask REST API           │
                      │  GET  /api/topics            │
                      │  POST /api/infer             │
                      │  GET  /api/stats             │
                      └──────────────┬──────────────┘
                                     ▼
                      ┌─────────────────────────────┐
                      │   Web Dashboard (TopicLens)  │
                      │  Live inference + explorer   │
                      └─────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Library | Version | Purpose |
|----------|---------|---------|---------|
| Core ML | scikit-learn | ≥1.3 | Data loading, TF-IDF, vectorizer |
| Classical NLP | gensim | ≥4.3 | LDA model, coherence scoring |
| Modern NLP | bertopic | ≥0.16 | BERTopic model |
| Embeddings | sentence-transformers | ≥2.6 | BERT sentence embeddings |
| Dim. reduction | umap-learn | ≥0.5 | UMAP dimensionality reduction |
| Clustering | hdbscan | ≥0.8 | Density-based clustering |
| Text tools | nltk | ≥3.8 | Stopwords, WordNet lemmatizer |
| Visualization | pyLDAvis | ≥3.4 | Interactive LDA plot |
| Visualization | matplotlib | ≥3.7 | Static plots and charts |
| Visualization | wordcloud | ≥1.9 | Topic word clouds |
| API | Flask + Flask-CORS | ≥3.0 | REST API |
| Production | gunicorn | ≥21 | WSGI server |
| Containers | Docker + Compose | latest | Deployment |
| Data | pandas / numpy | latest | Data manipulation |

---

## 📁 Project Structure

```
topic-modeling-nlp/
│
├── README.md                      ← This file
├── requirements.txt               ← Python dependencies
├── Dockerfile                     ← Container build instructions
├── docker-compose.yml             ← Multi-container orchestration
├── .gitignore
├── config.py                      ← All hyperparameters & settings
├── main.py                        ← 🚀 Full pipeline entry point
│
├── data/
│   ├── __init__.py
│   └── loader.py                  ← Load 20 Newsgroups or custom CSV
│
├── preprocessing/
│   ├── __init__.py
│   └── text_cleaner.py            ← Clean, tokenize, lemmatize
│
├── models/
│   ├── __init__.py
│   ├── lda_model.py               ← Gensim LDA wrapper (fit/infer/save/load)
│   └── bertopic_model.py          ← BERTopic wrapper (fit/infer/save/load)
│
├── evaluation/
│   ├── __init__.py
│   └── metrics.py                 ← Coherence, diversity, perplexity, sweep
│
├── visualization/
│   ├── __init__.py
│   └── plots.py                   ← pyLDAvis, coherence curve, word clouds, CSV
│
├── api/
│   ├── __init__.py
│   └── app.py                     ← Flask REST API (health, topics, infer, stats)
│
├── frontend/
│   └── index.html                 ← Dark-themed interactive dashboard
│
├── notebooks/
│   └── README.md                  ← Placeholder for Jupyter notebooks
│
└── outputs/                       ← Auto-created at runtime
    ├── saved_models/
    │   ├── lda.model
    │   ├── lda.dict
    │   ├── lda_corpus.pkl
    │   └── bertopic_model/
    ├── lda_vis.html
    ├── topic_words.png
    ├── topic_words.csv
    ├── coherence_scores.png
    └── wordclouds/
        ├── topic_0.png
        ├── topic_1.png
        └── ...
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip
- (Optional) GPU for faster BERTopic embedding

### Step 1 — Clone the Repository
```bash
git clone https://github.com/yourusername/topic-modeling-nlp.git
cd topic-modeling-nlp
```

### Step 2 — Create a Virtual Environment
```bash
python -m venv venv

# Activate
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Download NLTK Data
```bash
python -c "
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
"
```

### Step 5 — Verify Everything Works
```bash
python -c "
from bertopic import BERTopic
from gensim.models import LdaMulticore
from flask import Flask
print('✅ All dependencies OK')
"
```

---

## 🔄 Pipeline Walkthrough

### 1. Load Data
```python
from data.loader import load_20newsgroups

docs, labels, label_names = load_20newsgroups(subset='train')
# → 11,314 raw text documents
```

### 2. Preprocess
```python
from preprocessing.text_cleaner import preprocess

cleaned = preprocess(docs)
# → lowercased, URLs stripped, stopwords removed, lemmatized
# Example: "The NASA shuttle launch was delayed" → "nasa shuttle launch delay"
```

### 3. Train LDA
```python
from models.lda_model import LDATopicModel

lda = LDATopicModel(num_topics=20, passes=15)
lda.fit(cleaned)
print(lda.get_topics(num_words=10))
```

### 4. Train BERTopic
```python
from models.bertopic_model import BERTopicModel

bt = BERTopicModel(nr_topics='auto')
bt.fit(docs)                      # BERTopic uses raw docs
print(bt.get_topic_info())
```

### 5. Evaluate
```python
score = lda.coherence_score(cleaned, metric='c_v')
print(f"Coherence (Cv): {score:.4f}")
```

### 6. Visualize
```python
from visualization.plots import save_lda_vis, save_wordclouds

save_lda_vis(lda.model, lda.corpus, lda.dictionary)
save_wordclouds(lda.get_topics_dict())
```

### 7. Infer on New Text
```python
new_text = "NASA launched a new satellite to study climate change."
cleaned_new = preprocess([new_text])[0]
topic_dist = lda.infer(cleaned_new)
print(topic_dist)  # [(topic_id, probability), ...]
```

---

## 🤖 Models Used

### Model 1 — LDA (Latent Dirichlet Allocation)

**Algorithm:** Probabilistic generative model (variational Bayes via Gensim LdaMulticore).

**How it works:**
1. Represent each document as a **Bag-of-Words** (BoW) vector
2. Assume each document is a mixture of K topics
3. Assume each topic is a distribution over the vocabulary
4. Learn both distributions jointly using a Dirichlet prior
5. Use variational EM to approximate the posterior

**Key hyperparameters:**
```python
num_topics   = 20      # K — number of topics
passes       = 15      # training epochs over the entire corpus
alpha        = 'auto'  # document-topic sparsity prior (auto-tuned)
eta          = 'auto'  # topic-word sparsity prior (auto-tuned)
workers      = 2       # parallel CPU workers
```

**Pros:** Interpretable, fast, probabilistic topic assignments, well-studied  
**Cons:** BoW loses word order; sensitive to preprocessing quality; needs manual k selection

---

### Model 2 — BERTopic

**Algorithm:** Embedding → dimensionality reduction → density clustering → c-TF-IDF.

**How it works:**
1. **Embed** each document using `sentence-transformers` (all-MiniLM-L6-v2, 384-dim)
2. **Reduce** dimensions with UMAP (384 → 5 dims) to cluster-friendly space
3. **Cluster** with HDBSCAN (density-based; auto-determines cluster count)
4. **Extract** topic words using class-based TF-IDF (c-TF-IDF) per cluster

**Key hyperparameters:**
```python
embedding_model   = 'all-MiniLM-L6-v2'   # SentenceTransformer
umap_n_neighbors  = 15                    # UMAP neighborhood size
umap_n_components = 5                     # UMAP output dimensions
hdbscan_min_size  = 15                    # minimum cluster size
nr_topics         = 'auto'               # let HDBSCAN decide
```

**Pros:** Captures semantic meaning; no manual preprocessing needed; handles outliers (topic -1); usually better coherence  
**Cons:** Slower (embedding step ~5–10 min on CPU); harder to inspect probability distributions; non-deterministic

---

## 📊 Evaluation Metrics

### 1. Coherence Score Cv *(primary)*
Measures semantic similarity of top-N words within each topic using word co-occurrence in a sliding window.

```
Range   : 0 → 1   (higher is better)
Target  : > 0.45 acceptable · > 0.55 good · > 0.65 excellent
Formula : mean pairwise PMI of top-10 words per topic
```

### 2. UMass Coherence *(secondary)*
Based on document-level co-occurrence. Negative; closer to 0 is better.

```
Range   : −∞ → 0  (closer to 0 is better)
Target  : > −2.5
```

### 3. Topic Diversity
Fraction of unique words across top-10 words of all topics.  
Prevents degenerate solutions where every topic shares the same words.

```
Range   : 0 → 1   (higher = more diverse)
Target  : > 0.70
Formula : |unique words across all topics| / |total words across all topics|
```

### 4. Perplexity *(LDA only)*
How well the model predicts held-out documents.

```
Range   : 0 → ∞  (lower is better)
Note    : Can conflict with coherence — use coherence as primary metric
```

---

## 🌐 REST API Reference

Start the API server:
```bash
python api/app.py
# → http://localhost:5000
```

### Endpoints

#### `GET /api/health`
Check if server and model are ready.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

#### `GET /api/topics`
Return all discovered topics with their top words.

**Response:**
```json
{
  "num_topics": 20,
  "topics": [
    { "topic_id": 0, "words": ["space", "nasa", "orbit", "shuttle", "launch", "satellite"] },
    { "topic_id": 1, "words": ["game", "team", "player", "season", "win", "goal"] }
  ]
}
```

---

#### `POST /api/infer`
Infer topic distribution for a given text.

**Request body:**
```json
{ "text": "NASA launched a new rocket to the International Space Station." }
```

**Response:**
```json
{
  "input_text": "NASA launched a new rocket...",
  "topics": [
    { "topic_id": 2, "probability": 0.8421, "words": ["space", "nasa", "orbit"] },
    { "topic_id": 7, "probability": 0.0912, "words": ["science", "research", "study"] }
  ]
}
```

---

#### `GET /api/topic/<id>`
Get detailed word list for a single topic.

**Response:**
```json
{ "topic_id": 3, "words": ["window", "file", "program", "software", "dos", "memory"] }
```

---

#### `GET /api/stats`
Return model metadata.

**Response:**
```json
{
  "num_topics": 20,
  "vocab_size": 12483,
  "corpus_size": 11223,
  "passes": 15,
  "alpha": "auto"
}
```

---

## 🖥️ Web Dashboard

The dashboard (`frontend/index.html`) is served at `http://localhost:5000` and provides:

- **Stats panel** — topics, vocab size, documents, passes
- **Topic explorer** — click any topic to see all keywords with visual weight bars
- **Live inference** — paste any text, click Analyze → see topic probability distribution
- **Sample texts** — one-click presets for Space, Religion, Tech, Sports, Politics
- **Keyboard shortcut** — `Ctrl+Enter` to run inference

The dashboard talks to the Flask API on the same origin. To run it standalone during development:
```bash
# Terminal 1 — API
python api/app.py

# Terminal 2 — open browser
open http://localhost:5000
```

---

## 🐳 Docker Deployment

### Option A — Docker Compose (recommended)

```bash
# Build and start
docker-compose up --build

# Train the model inside the container (first time only)
docker exec topic-modeling-nlp python main.py --model lda

# Dashboard is now live at:
# http://localhost:5000
```

### Option B — Docker CLI

```bash
# Build image
docker build -t topiclens .

# Run container
docker run -d \
  --name topiclens \
  -p 5000:5000 \
  -v $(pwd)/outputs:/app/outputs \
  topiclens

# Train inside container
docker exec topiclens python main.py --model lda

# Open dashboard
open http://localhost:5000
```

### Option C — Cloud Deployment (Render / Railway / Fly.io)

1. Push to GitHub
2. Connect your repo on Render/Railway
3. Set **Start command:** `gunicorn -b 0.0.0.0:$PORT --timeout 120 api.app:app`
4. Add a build step: `python main.py --no-vis` to train on startup
5. Done — your API is live at `https://your-app.render.com`

---

## 🚀 How to Run

### Full Pipeline (LDA + all outputs)
```bash
python main.py
```

### LDA Only
```bash
python main.py --model lda
```

### BERTopic Only
```bash
python main.py --model bertopic
```

### Both Models
```bash
python main.py --model both
```

### Find Optimal Number of Topics
```bash
python main.py --model lda --tune
# Sweeps k = 5, 10, 15, 20, 25, 30
# Saves coherence curve → outputs/coherence_scores.png
```

### Override Topic Count
```bash
python main.py --model lda --topics 15
```

### Skip Visualizations (faster)
```bash
python main.py --no-vis
```

### Start the API Server
```bash
python api/app.py
# → http://localhost:5000
```

### Production API with Gunicorn
```bash
gunicorn -b 0.0.0.0:5000 --workers 2 --timeout 120 api.app:app
```

---

## 📈 Expected Results

### LDA (20 topics, 15 passes, 20 Newsgroups train set)

| Metric | Expected Score |
|--------|---------------|
| Coherence Cv | 0.48 – 0.56 |
| UMass Coherence | −1.8 to −2.5 |
| Topic Diversity | 0.70 – 0.80 |
| Perplexity | −7.5 to −9.0 |
| Training Time (CPU) | ~3–5 min |

### BERTopic (auto topics, same corpus)

| Metric | Expected Score |
|--------|---------------|
| Topics Found | 18 – 25 |
| Coherence Cv | 0.52 – 0.62 |
| Topic Diversity | 0.78 – 0.88 |
| Training Time (CPU) | ~8–15 min |
| Training Time (GPU) | ~2–4 min |

### Sample Discovered Topics (LDA)

```
Topic  0 → space nasa orbit shuttle launch satellite mission earth probe planet
Topic  1 → game team player season win score hockey play goal league
Topic  2 → god religion christian church bible believe faith jesus moral sin
Topic  3 → drive disk scsi controller ide hard mb interface speed card
Topic  4 → window file program software dos memory run install error version
Topic  5 → gun weapon firearm law crime police government control right amendment
Topic  6 → image graphic color bit format display screen pixel resolution software
Topic  7 → key encryption message public algorithm security private system data code
Topic  8 → car engine speed road drive model buy price dealer year
Topic  9 → medical health disease patient treatment doctor study drug symptom hospital
```

---

## 🎨 Visualizations

| Output | Path | Description |
|--------|------|-------------|
| Interactive LDA | `outputs/lda_vis.html` | pyLDAvis — inter-topic distance map + term explorer |
| Topic word chart | `outputs/topic_words.png` | Horizontal bar charts of top words for first 9 topics |
| Coherence curve | `outputs/coherence_scores.png` | Cv score vs number of topics (for tuning) |
| Word clouds | `outputs/wordclouds/topic_N.png` | One word cloud image per topic |
| Topic words CSV | `outputs/topic_words.csv` | Structured table: topic_id, rank, word |

Open the pyLDAvis visualization:
```bash
open outputs/lda_vis.html        # macOS
xdg-open outputs/lda_vis.html   # Linux
start outputs/lda_vis.html       # Windows
```

---

## 🔧 Configuration

All hyperparameters live in `config.py` — no code changes needed for experimentation:

```python
# config.py

# Data
DATASET_SUBSET   = "train"       # 'train' | 'test' | 'all'
CATEGORIES       = None          # None = all 20 categories

# Preprocessing
MIN_TOKEN_LENGTH = 3
MAX_DF           = 0.40          # drop tokens in >40% of docs
MIN_DF           = 10            # drop tokens in <10 docs

# LDA
LDA_NUM_TOPICS   = 20
LDA_PASSES       = 15
LDA_ALPHA        = "auto"        # document-topic density
LDA_ETA          = "auto"        # topic-word density

# BERTopic
BERT_EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
BERT_NR_TOPICS         = "auto"
BERT_UMAP_NEIGHBORS    = 15
BERT_UMAP_COMPONENTS   = 5
BERT_HDBSCAN_MIN_SIZE  = 15

# Evaluation
COHERENCE_METRIC       = "c_v"
TOP_WORDS_PER_TOPIC    = 10
TUNE_TOPICS_RANGE      = range(5, 35, 5)

# API
API_HOST         = "0.0.0.0"
API_PORT         = 5000
```

---

## 🔮 Future Improvements

| Feature | Description |
|---------|-------------|
| Dynamic Topic Modeling | Track topic evolution over time using `gensim.models.LdaSeqModel` |
| Guided Topics | Seed topics with domain keywords using Contextualized Topic Models (CTM) |
| Multilingual | Swap to `paraphrase-multilingual-MiniLM-L12-v2` for multi-language corpora |
| Online LDA | Incremental learning on streaming data with `gensim.models.ldamodel` online mode |
| Hierarchical Topics | Use `BERTopic.hierarchical_topics()` to build topic trees |
| GPU Acceleration | Add RAPIDS cuML for GPU-accelerated UMAP + HDBSCAN |
| Streamlit Dashboard | Rich Streamlit app with topic evolution charts and document search |
| Auth + Rate Limiting | Add JWT auth and rate limiting to the Flask API for production hardening |
| CI/CD Pipeline | GitHub Actions for automated testing and Docker image publishing |

---

## 📚 References

1. Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). *Latent Dirichlet Allocation*. Journal of Machine Learning Research, 3, 993–1022.
2. Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure*. arXiv:2203.05794.
3. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019.
4. McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection*. arXiv:1802.03426.
5. Campello, R., Moulavi, D., & Sander, J. (2013). *HDBSCAN: Density-Based Clustering*. PAKDD 2013.
6. [20 Newsgroups Dataset](http://qwone.com/~jason/20Newsgroups/) — Jason Rennie
7. [Gensim LDA Documentation](https://radimrehurek.com/gensim/models/ldamodel.html)
8. [BERTopic Documentation](https://maartengr.github.io/BERTopic/)
9. [pyLDAvis Documentation](https://pyldavis.readthedocs.io/)

---

## 👨‍💻 Author

**Your Name**
ML Engineer · NLP Enthusiast
[GitHub](https://github.com/yourusername) · [LinkedIn](https://linkedin.com/in/yourprofile)

---

> ⭐ If this project helped you, consider giving it a star on GitHub!

---

*MIT License — free to use, modify, and distribute.*
