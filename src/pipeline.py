# src/pipeline.py

import pandas as pd
from .retrieval import retrieve_reviews, get_cluster_distribution
from .insight_generation import generate_insight
from .summary import summarize_reviews

import re
from bs4 import BeautifulSoup
import os


# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLUSTERS_CSV = os.path.join(BASE_DIR, "analysis", "reviews_with_clusters.csv")
CLUSTER_KEYWORDS_CSV = os.path.join(BASE_DIR, "analysis", "cluster_keywords.csv")

clusters_df = pd.read_csv(CLUSTERS_CSV)
cluster_keywords_df = pd.read_csv(CLUSTER_KEYWORDS_CSV)


# -------------------------------
# Runtime Cleaning
# -------------------------------
def clean_text_runtime(text):
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------------------------------
# Main Pipeline
# -------------------------------
def full_pipeline(query, k=10):

    results = retrieve_reviews(query, k=k)

    top_reviews = []
    indices = []

    for item in results:
        clean_text = clean_text_runtime(item["review_text"])

        top_reviews.append({
            "review_text": item["review_text"],
            "clean_text": clean_text,
            "rating": item.get("rating", 3)
        })

        indices.append(item["index"])

    # Cluster distribution
    cluster_info = get_cluster_distribution(indices, clusters_df)

    distribution = cluster_info.get("distribution", {})

    if not distribution:
        dominant_cluster = 0
    else:
        dominant_cluster = max(distribution, key=distribution.get)

    # Extract keywords correctly
    keywords_series = cluster_keywords_df[
        cluster_keywords_df["cluster"] == dominant_cluster
    ]["keywords"]

    if not keywords_series.empty:
        raw_keywords = keywords_series.iloc[0]
        cluster_keywords = [k.strip() for k in raw_keywords.split(",") if k.strip()]
    else:
        cluster_keywords = []

    # Generate insight
    insight = generate_insight(
        top_reviews=top_reviews,
        cluster_info=cluster_info,
        cluster_keywords=cluster_keywords
    )

    # Summarization
    clean_texts = [r["clean_text"] for r in top_reviews]
    summary = summarize_reviews(clean_texts)

    return {
        "query": query,
        "top_reviews": top_reviews,
        "cluster_info": cluster_info,
        "insight": insight,
        "summary": summary
    }