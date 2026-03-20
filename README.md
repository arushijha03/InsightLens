# InsightLens — Amazon Review Intelligence Platform

**Turn thousands of raw product reviews into structured, actionable business insights — powered by semantic search, clustering, and NLP.**

InsightLens is an end-to-end NLP pipeline that ingests Amazon product reviews, embeds them into a vector space, clusters them by topic, and surfaces structured insights (themes, strengths, pain points, recommendations) through an interactive Streamlit dashboard — all without relying on LLMs.

[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0467DF?logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
---

## Key Features

- **Semantic Retrieval** — Query reviews in natural language; FAISS returns the most relevant matches via cosine similarity
- **Automatic Clustering** — PCA + KMeans groups 200K+ reviews into 50 topic clusters with TF-IDF keyword extraction
- **Structured Insights** — Surfaces dominant themes, product strengths, customer pain points, and business recommendations
- **Extractive Summarization** — Generates concise summaries using TF-IDF sentence ranking — no LLM needed
- **Interactive Dashboard** — Streamlit UI with sentiment charts, word clouds, and side-by-side insight panels
- **Batch Processing** — Run multiple queries at once and export results to JSON

---

## Architecture

```
                         ┌─────────────────────────────────────────┐
                         │            OFFLINE PIPELINE              │
                         │                                         │
  Amazon Reviews (568K)  │  Preprocess → Embed → FAISS Index       │
          CSV            │                 ↓                       │
                         │         PCA → KMeans (k=50)             │
                         │                 ↓                       │
                         │     TF-IDF Keyword Extraction           │
                         └──────────────────┬──────────────────────┘
                                            │
                         ┌──────────────────▼──────────────────────┐
                         │           RUNTIME PIPELINE               │
                         │                                         │
  User Query ──→ Embed ──┤──→ FAISS Top-K Retrieval                │
                         │        ↓                                │
                         │   Cluster Distribution Analysis         │
                         │        ↓                                │
                         │   Insight Generation (TF-IDF + Rules)   │
                         │        ↓                                │
                         │   Extractive Summarization              │
                         └──────────────────┬──────────────────────┘
                                            │
                                            ▼
                                   Streamlit Dashboard
                           (Reviews • Insights • Visualizations)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| **Vector Search** | FAISS `IndexFlatIP` with L2 normalization |
| **Clustering** | PCA (50 components) + KMeans (k=50) |
| **Keyword Extraction** | TF-IDF (1–2 grams, custom stopwords) |
| **Summarization** | Extractive — TF-IDF sentence ranking + overlap deduplication |
| **Frontend** | Streamlit (wide layout, interactive) |
| **Data Processing** | Pandas, NumPy, scikit-learn |

---

## Project Structure

```
InsightLens/
├── app.py                          # Streamlit dashboard
├── requirements.txt
├── src/
│   ├── pipeline.py                 # End-to-end runtime pipeline
│   ├── preprocess.py               # Data cleaning & filtering
│   ├── embedder.py                 # Sentence embedding generation
│   ├── build_index.py              # FAISS index construction
│   ├── retrieval.py                # Semantic search & cluster distribution
│   ├── clustering.py               # PCA + KMeans clustering
│   ├── theme_extraction.py         # TF-IDF keywords per cluster
│   ├── insight_generation.py       # Strengths, pain points, recommendations
│   ├── summary.py                  # Extractive summarization
│   └── visualization.py            # Sentiment charts & word clouds
├── notebooks/
│   ├── week1_pipeline.ipynb        # Data exploration (568K reviews)
│   ├── week2_pipeline.ipynb        # Embeddings, PCA, KMeans
│   ├── week3_pipeline.ipynb        # Retrieval + insight generation
│   ├── week4_pipeline.ipynb        # FAISS index experiments
│   ├── week5_pipeline.ipynb        # Cluster keyword inspection
│   └── week6_pipeline.ipynb        # Full pipeline end-to-end
├── reports/
│   ├── evaluation_metrics.json     # Retrieval: P@10, R@10, nDCG@10
│   └── insight_evaluation.json     # Actionability & coverage scores
├── faiss_index/                    # Pre-built FAISS index
├── embeddings/                     # Embeddings + review mapping
└── analysis/                       # Cluster assignments & keywords
```

---

## Evaluation Results

| Metric | Score |
|--------|-------|
| Precision@10 | **0.80** |
| Recall@10 | **0.70** |
| nDCG@10 | **0.75** |
| Actionability Score | **0.90** |
| Coverage Score | **0.95** |
| Redundancy Score | **0.10** (lower is better) |

---

## Getting Started

### Prerequisites

- Python 3.10+

### Installation

```bash
git clone https://github.com/arushijha03/InsightLens.git
cd InsightLens
pip install -r requirements.txt
```

### Run the Dashboard

```bash
python -m streamlit run app.py
```

The app opens at `http://localhost:8501`. Enter a query like *"Top complaints about coffee taste?"* and click **Run Pipeline**.

---

## How It Works

1. **Preprocessing** — Raw Amazon reviews (568K) are cleaned, filtered (min 20 words), and sampled to 200K
2. **Embedding** — Each review is encoded into a 384-dim vector using `all-MiniLM-L6-v2`
3. **Indexing** — Vectors are L2-normalized and stored in a FAISS inner-product index for fast retrieval
4. **Clustering** — PCA reduces dimensionality to 50, then KMeans assigns reviews to 50 topic clusters
5. **Theme Extraction** — TF-IDF extracts top keywords per cluster for interpretability
6. **Query Time** — User query is embedded → FAISS retrieves top-k reviews → cluster distribution is analyzed → insights and summaries are generated
7. **Visualization** — Sentiment distribution, word clouds, and structured insight panels are displayed in Streamlit

---

## Sample Query Output

**Query:** *"Top complaints about coffee taste?"*

| Section | Output |
|---------|--------|
| **Dominant Theme** | flavor, beans, taste |
| **Strengths** | robust, likes, best |
| **Pain Points** | disgusted, sharp, metallic |
| **Recommendation** | Monitor customer feedback trends to identify improvement areas |
| **Avg Rating** | 4.0 |
| **Sentiment** | 60% positive, 20% negative, 20% neutral |

---

## Future Improvements

- [ ] Add LLM-powered abstractive summarization as an optional mode
- [ ] Implement comparative analysis across product categories
- [ ] Add temporal trend analysis for reviews over time
- [ ] Deploy on Hugging Face Spaces with full data
