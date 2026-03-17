import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib


def load_embeddings(path):
    embeddings = np.load(path)
    embeddings = embeddings.astype("float32")
    print(f"Loaded embeddings shape: {embeddings.shape}")
    return embeddings


def apply_pca(embeddings, n_components=50):
    print("Applying PCA...")
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    print(f"Reduced shape: {reduced.shape}")
    return reduced, pca


def run_kmeans(embeddings, n_clusters=50):
    print(f"Running KMeans with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans


def save_model(model, path):
    joblib.dump(model, path)