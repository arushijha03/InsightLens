import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

index = faiss.read_index("faiss_index/amazon_reviews_index.faiss")

with open("embeddings/review_id_mapping.json") as f:
    mapping = json.load(f)


def search_reviews(query, k=5):

    query_emb = model.encode([query])

    faiss.normalize_L2(query_emb)

    scores, ids = index.search(query_emb, k)

    results = []

    for idx in ids[0]:

        results.append(mapping[str(idx)])

    return results


if __name__ == "__main__":

    query = "coffee tastes bitter"

    results = search_reviews(query)

    for r in results:

        print("\nCategory:", r["category"])
        print("Rating:", r["rating"])
        print("Review:", r["review_text"])