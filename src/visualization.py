import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import json

def plot_sentiment(sentiment_data):
    """
    Plot positive vs negative ratio and average rating
    """
    labels = ['Positive', 'Negative', 'Neutral']
    ratios = [
        sentiment_data.get('positive_ratio', 0),
        sentiment_data.get('negative_ratio', 0),
        1 - sentiment_data.get('positive_ratio',0) - sentiment_data.get('negative_ratio',0)
    ]

    plt.figure(figsize=(6,4))
    plt.bar(labels, ratios, color=['green','red','gray'])
    plt.title(f"Sentiment Distribution (Avg Rating: {sentiment_data.get('avg_rating', 0):.2f})")
    plt.ylabel("Ratio")
    plt.show()


def plot_cluster_sizes(df):
    """
    Plot number of reviews per cluster
    """
    cluster_counts = df['cluster'].value_counts().sort_index()
    plt.figure(figsize=(8,4))
    cluster_counts.plot(kind='bar', color='skyblue')
    plt.title("Number of Reviews per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.show()


def generate_wordcloud(keywords_text, title="Keywords WordCloud"):
    """
    Generate a word cloud for top keywords.
    Skip if keywords_text is empty.
    """
    if not keywords_text.strip():
        print(f"⚠️ Skipping WordCloud for '{title}' — no keywords provided.")
        return

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(keywords_text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

def visualize_final_report(report_json_path):
    """
    Full visualizations from the final report JSON
    """
    with open(report_json_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    # Sentiment
    plot_sentiment(report.get('sentiment', {}))

    # WordClouds
    dominant_keywords = " ".join(report.get('dominant_theme', []))
    strengths_keywords = " ".join(report.get('strengths', []))
    pain_points_keywords = " ".join(report.get('pain_points', []))

    generate_wordcloud(dominant_keywords, title="Dominant Theme Keywords")
    generate_wordcloud(strengths_keywords, title="Strengths Keywords")
    generate_wordcloud(pain_points_keywords, title="Pain Points Keywords")