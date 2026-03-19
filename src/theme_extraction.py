import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def extract_keywords_per_cluster(df, cluster_labels, top_k=10):
    """
    Extract top-k keywords per cluster using TF-IDF.
    """
    print("Extracting keywords per cluster...")
    df["cluster"] = cluster_labels
    results = []

    for cluster_id in sorted(df["cluster"].unique()):
        print(f"Processing cluster {cluster_id}...")
        texts = df[df["cluster"] == cluster_id]["clean_text"]

        custom_stopwords = [
            "good", "great", "like", "love", "product", "buy", "just",
            "really", "nice", "best", "better", "well", "also"
        ]
        stop_words = list(ENGLISH_STOP_WORDS.union(custom_stopwords))

        vectorizer = TfidfVectorizer(
            max_features=3000,
            stop_words=stop_words,
            ngram_range=(1,2),
            min_df=10,
            max_df=0.6
        )

        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.mean(axis=0).A1
        top_indices = scores.argsort()[::-1][:top_k]
        keywords = [feature_names[i] for i in top_indices]

        results.append({
            "cluster": cluster_id,
            "keywords": ", ".join(keywords)
        })

    return pd.DataFrame(results)

def generate_insight(reviews, cluster_info, cluster_keywords_df):
    """
    Generate structured insight from retrieved reviews and cluster info.
    """
    dominant_cluster = cluster_info["dominant_cluster"]
    keywords_row = cluster_keywords_df[cluster_keywords_df["cluster"] == dominant_cluster]
    keywords = keywords_row["keywords"].values[0] if not keywords_row.empty else ""

    insight_text = f"Dominant cluster {dominant_cluster} with keywords: {keywords}"

    return {
        "dominant_cluster": dominant_cluster,
        "keywords": keywords,
        "insight_text": insight_text
    }