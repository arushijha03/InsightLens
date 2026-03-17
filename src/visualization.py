import umap
import matplotlib.pyplot as plt


def plot_umap(embeddings, labels, save_path):
    print("Running UMAP for visualization...")

    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, s=5)
    plt.title("UMAP Cluster Visualization")

    plt.savefig(save_path)
    plt.close()

    print(f"Saved plot to {save_path}")