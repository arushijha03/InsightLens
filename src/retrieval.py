import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index", "amazon_reviews_index.faiss")
MAPPING_PATH = os.path.join(BASE_DIR, "embeddings", "review_id_mapping.json")

print("USING FIXED PATH VERSION")
print("INDEX PATH:", INDEX_PATH)
print("MAPPING PATH:", MAPPING_PATH)

# Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index(INDEX_PATH)

# Load mapping
with open(MAPPING_PATH) as f:
    mapping = json.load(f)

# Ensure all keys are str
mapping = {str(k): v for k, v in mapping.items()}

def embed_query(query: str) -> np.ndarray:
    """Embed a query string using the same SentenceTransformer model."""
    query_emb = model.encode([query])
    faiss.normalize_L2(query_emb)
    return query_emb

def retrieve_reviews(query: str, k: int = 10):
    """
    Retrieve top-k reviews using FAISS.
    Returns list of dicts:
        { "index": idx, "review_text": ..., "product_id": ..., "category": ..., "rating": ..., "score": ... }
    """
    query_emb = embed_query(query)
    scores, ids = index.search(query_emb, k)

    results = []
    for i, idx in enumerate(ids[0]):
        review_dict = mapping[str(idx)].copy()
        review_dict["index"] = idx
        review_dict["score"] = float(scores[0][i])
        results.append(review_dict)

    return results

def get_cluster_distribution(indices, df):
    """
    Compute dominant cluster and cluster frequencies for a list of review indices.
    """
    clusters = df.iloc[indices]["cluster"].tolist()
    cluster_counts = {}
    for c in clusters:
        cluster_counts[c] = cluster_counts.get(c, 0) + 1

    dominant_cluster = max(cluster_counts, key=cluster_counts.get)
    top_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        "dominant_cluster": dominant_cluster,
        "cluster_counts": cluster_counts,
        "top_clusters": top_clusters
    }