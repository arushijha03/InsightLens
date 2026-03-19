import pandas as pd
from .retrieval import retrieve_reviews, get_cluster_distribution
from .theme_extraction import generate_insight
from .summary import summarize_reviews
import re
from bs4 import BeautifulSoup

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
CLUSTERS_CSV = os.path.join(BASE_DIR, "analysis", "reviews_with_clusters.csv")
CLUSTER_KEYWORDS_CSV = os.path.join(BASE_DIR, "analysis", "cluster_keywords.csv")

clusters_df = pd.read_csv(CLUSTERS_CSV)
cluster_keywords_df = pd.read_csv(CLUSTER_KEYWORDS_CSV)


def clean_text_runtime(text):
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def full_pipeline(query, k=10):
    """
    Orchestrated pipeline: Query → Retrieval → Cluster → Insight → Summary
    Compatible with week3_pipeline.ipynb interface.
    
    Args:
        query (str): Raw query string
        k (int): Number of top reviews to retrieve
    Returns:
        dict: Structured output including reviews, clusters, insight, summary
    """
    # 1️⃣ Retrieve top reviews
    results = retrieve_reviews(query, k=k)  # list of dicts

    # 2️⃣ Extract indices and review texts
    indices = [item["index"] for item in results]
    retrieved_reviews = [clean_text_runtime(item["review_text"]) for item in results]

    # 3️⃣ Cluster-aware distribution
    cluster_info = get_cluster_distribution(indices, clusters_df)

    # 4️⃣ Generate structured insight
    insight = generate_insight(retrieved_reviews, cluster_info, cluster_keywords_df)

    # 5️⃣ Summarize reviews
    summary = summarize_reviews(retrieved_reviews)

    # 6️⃣ Package output
    output = {
        "query": query,
        "top_reviews": retrieved_reviews,
        "cluster_info": cluster_info,
        "insight": insight,
        "summary": summary
    }

    return output