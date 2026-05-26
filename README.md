# InsightLens — Amazon Review Intelligence Platform

**Turn thousands of raw product reviews into structured, actionable business insights — powered by semantic search, clustering, NLP, and LLMs.**

[![Live Demo](https://img.shields.io/badge/Live_Demo-InsightLens-FF4B4B?logo=streamlit&logoColor=white)](https://euec5wxonk9zyfm4w2wzht.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0467DF?logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai&logoColor=white)](https://openai.com)

**[Try the Live Demo](https://euec5wxonk9zyfm4w2wzht.streamlit.app/)**

---

## Problem Statement

E-commerce platforms generate millions of product reviews, but extracting actionable intelligence from this unstructured text at scale remains a challenge. Product managers, brand analysts, and marketing teams need to quickly understand:

- **What do customers love?** — Identify product strengths driving positive sentiment
- **What are the pain points?** — Surface recurring complaints before they escalate
- **What should we do about it?** — Generate data-driven business recommendations

Manual analysis of even a few hundred reviews is time-consuming and subjective. Keyword search misses semantic meaning ("terrible flavor" vs "taste is awful"), and traditional dashboards only show aggregate ratings without explaining *why* customers feel the way they do.

**InsightLens** solves this by combining **semantic search**, **unsupervised clustering**, and **LLM-powered analysis** into a single pipeline that turns 568K+ raw Amazon reviews into structured, query-driven insights — delivered through an interactive dashboard in seconds.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         OFFLINE PIPELINE                             │
│                                                                      │
│  Amazon Reviews (568K)                                               │
│        │                                                             │
│        ▼                                                             │
│  ┌───────────┐    ┌──────────────────┐    ┌────────────────────┐     │
│  │ Preprocess │───▶│ Sentence Embedder │───▶│ FAISS Index Builder │    │
│  │ Clean/Filter│   │ MiniLM-L6-v2     │    │ IndexFlatIP (IP)   │    │
│  │ 200K sample │   │ 384-dim vectors  │    │ L2 normalized      │    │
│  └───────────┘    └──────────────────┘    └────────────────────┘     │
│        │                                                             │
│        ▼                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐        │
│  │ PCA (50) │───▶│ KMeans (k=50)│───▶│ TF-IDF Theme Keywords │       │
│  └──────────┘    └──────────────┘    └──────────────────────┘        │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         RUNTIME PIPELINE                             │
│                                                                      │
│  User Query                                                          │
│      │                                                               │
│      ▼                                                               │
│  ┌──────────────────┐    ┌──────────────────────┐                    │
│  │ Query Embedding   │───▶│ FAISS Top-K Retrieval │                   │
│  │ MiniLM-L6-v2     │    │ Cosine Similarity     │                   │
│  └──────────────────┘    └──────────┬───────────┘                    │
│                                     │                                │
│                                     ▼                                │
│                          ┌────────────────────┐                      │
│                          │ Cluster Distribution│                      │
│                          │ Analysis            │                      │
│                          └────────┬───────────┘                      │
│                                   │                                  │
│                    ┌──────────────┼──────────────┐                   │
│                    ▼              │              ▼                    │
│  ┌──────────────────────┐        │  ┌──────────────────────┐         │
│  │  LLM Insights        │        │  │ Extractive Summary   │         │
│  │  (GPT-4o-mini)       │        │  │ TF-IDF Sentence      │         │
│  │        │              │        │  │ Ranking + Dedup      │         │
│  │        ▼ (on failure) │        │  └──────────────────────┘         │
│  │  TF-IDF Fallback     │        │                                   │
│  └──────────────────────┘        │                                   │
│                    │              │                                   │
│                    └──────────────┘                                   │
│                           │                                          │
│                           ▼                                          │
│               ┌──────────────────────┐                               │
│               │  Streamlit Dashboard  │                               │
│               │  Toggle: LLM / TF-IDF│                               │
│               │  Charts & Word Clouds │                               │
│               └──────────────────────┘                               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Encode reviews into 384-dim semantic vectors |
| **Vector Search** | FAISS `IndexFlatIP` | Sub-second cosine similarity retrieval over 200K vectors |
| **Clustering** | PCA (50 components) + KMeans (k=50) | Group reviews into coherent topic clusters |
| **Insight Generation** | OpenAI GPT-4o-mini (primary) | LLM-powered themes, strengths, pain points, recommendations |
| **Insight Fallback** | TF-IDF + rule-based engine | Automatic fallback when LLM is unavailable |
| **Keyword Extraction** | TF-IDF (1–2 grams, custom stopwords) | Extract interpretable keywords per cluster |
| **Summarization** | Extractive — TF-IDF sentence ranking | Concise summaries with Jaccard overlap deduplication |
| **Frontend** | Streamlit | Interactive dashboard with toggle, charts, and word clouds |
| **Data Processing** | Pandas, NumPy, scikit-learn | Cleaning, filtering, and statistical operations |
| **Environment** | python-dotenv | Secure API key management via `.env` |

---

## Key Features

- **Semantic Retrieval** — Query reviews in natural language; FAISS returns the most relevant matches via cosine similarity
- **Automatic Clustering** — PCA + KMeans groups 200K+ reviews into 50 topic clusters with TF-IDF keyword extraction
- **LLM-Powered Insights** — GPT-4o-mini generates dominant themes, strengths, pain points, and business recommendations
- **Graceful Fallback** — If the LLM is unavailable or toggled off, TF-IDF-based insight generation kicks in automatically
- **UI Toggle** — Switch between LLM and TF-IDF insight generation directly from the dashboard sidebar
- **Extractive Summarization** — Short and detailed summaries using TF-IDF sentence ranking
- **Interactive Dashboard** — Streamlit UI with sentiment charts, word clouds, and side-by-side insight panels
- **Batch Processing** — Run multiple queries at once and export results to JSON

---

## Business Impact

| Impact Area | Description |
|-------------|-------------|
| **Faster Decision-Making** | Reduces manual review analysis from hours/days to seconds per query across 568K reviews |
| **Customer Pain Point Detection** | Surfaces recurring complaints (e.g., taste, packaging, freshness) that may not appear in aggregate star ratings |
| **Product Improvement Signals** | Generates specific, actionable recommendations (e.g., "Improve packaging durability") tied directly to customer feedback |
| **Marketing Intelligence** | Identifies what customers value most (strengths) to inform ad copy, product descriptions, and positioning |
| **Scalable Monitoring** | Batch query mode enables teams to track multiple product dimensions simultaneously and export findings for reporting |
| **Cost-Efficient Analysis** | LLM insights via GPT-4o-mini keep per-query costs minimal (~$0.001/query); TF-IDF fallback ensures zero-cost operation when needed |

---

## Evaluation Results

### Retrieval Quality

| Metric | Score | What It Means |
|--------|-------|---------------|
| **Precision@10** | **0.80** | 8 out of 10 retrieved reviews are relevant to the query |
| **Recall@10** | **0.70** | 70% of all relevant reviews are captured in the top 10 |
| **nDCG@10** | **0.75** | The most relevant reviews are ranked near the top of results |

### Insight Quality

| Metric | Score | What It Means |
|--------|-------|---------------|
| **Actionability** | **0.90** | 90% of generated insights contain actionable recommendations |
| **Coverage** | **0.95** | 95% of key topics in the reviews are represented in insights |
| **Redundancy** | **0.10** | Only 10% overlap between insight points (lower is better) |

*Evaluated across 5 benchmark queries: bad coffee taste, dog treats too hard, spicy flavor complaints, sweetness too low in candy, coffee aroma weak.*

---

## Getting Started

### Live Demo

| | Link |
|---|------|
| Try the app (Streamlit UI) | [https://euec5wxonk9zyfm4w2wzht.streamlit.app/](https://euec5wxonk9zyfm4w2wzht.streamlit.app/) |

**Quick try:** open the Streamlit app → enter a query (e.g. *"Top complaints about coffee taste?"*) → toggle LLM on/off → click **Run Pipeline**.

### Run Locally

#### Prerequisites

- Python 3.10+
- OpenAI API key (optional — TF-IDF fallback works without it)

#### 1. Clone the Repository

```bash
git clone https://github.com/arushijha03/InsightLens.git
cd InsightLens
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Configure LLM (Optional)

```bash
cp .env.example .env
```

Open `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-key-here
```

Get a key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys).

> If no API key is set, the system automatically uses TF-IDF-based insight generation. You can also toggle between LLM and TF-IDF from the dashboard sidebar at runtime.

#### 4. Run the Dashboard

```bash
python -m streamlit run app.py
```

The app opens at `http://localhost:8501`. Enter a query like *"Top complaints about coffee taste?"* and click **Run Pipeline**.

---

## How It Works

1. **Preprocessing** — Raw Amazon reviews (568K) are cleaned, HTML-stripped, filtered (min 20 words), and sampled to 200K
2. **Embedding** — Each review is encoded into a 384-dim vector using `all-MiniLM-L6-v2` in batches of 64
3. **Indexing** — Vectors are L2-normalized and stored in a FAISS inner-product index for fast retrieval
4. **Clustering** — PCA reduces dimensionality to 50, then KMeans assigns reviews to 50 topic clusters
5. **Theme Extraction** — TF-IDF extracts top keywords per cluster for interpretability
6. **Query Time** — User query is embedded → FAISS retrieves top-k reviews → cluster distribution is analyzed → GPT-4o-mini generates insights (falls back to TF-IDF if unavailable or toggled off) → extractive summaries are generated
7. **Visualization** — Sentiment distribution, word clouds, and structured insight panels are displayed in Streamlit

---

## Project Structure

```
InsightLens/
├── app.py                          # Streamlit dashboard (LLM/TF-IDF toggle)
├── requirements.txt                # Python dependencies
├── .env.example                    # API key template
├── .gitignore
├── src/
│   ├── pipeline.py                 # End-to-end runtime pipeline
│   ├── preprocess.py               # Data cleaning & filtering
│   ├── embedder.py                 # Sentence embedding generation
│   ├── build_index.py              # FAISS index construction
│   ├── retrieval.py                # Semantic search & cluster distribution
│   ├── clustering.py               # PCA + KMeans clustering
│   ├── theme_extraction.py         # TF-IDF keywords per cluster
│   ├── insight_generation.py       # LLM + TF-IDF insight generation
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

## Sample Query Output

**Query:** *"Top complaints about coffee taste?"*

| Section | Output |
|---------|--------|
| **Insight Source** | GPT-4o-mini (or TF-IDF if toggled off) |
| **Dominant Theme** | flavor, beans, taste |
| **Strengths** | robust, likes, best |
| **Pain Points** | disgusted, sharp, metallic |
| **Recommendation** | Monitor customer feedback trends to identify improvement areas |
| **Avg Rating** | 4.0 |
| **Sentiment** | 60% positive, 20% negative, 20% neutral |
