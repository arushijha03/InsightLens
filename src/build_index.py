import numpy as np
import faiss


def build_index():

    embeddings = np.load("embeddings/amazon_embeddings.npy")

    dimension = embeddings.shape[1]

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    faiss.write_index(index, "faiss_index/amazon_reviews_index.faiss")

    print("FAISS index saved!")


if __name__ == "__main__":

    build_index()