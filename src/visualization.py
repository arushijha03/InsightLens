import streamlit as st
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
        1 - sentiment_data.get('positive_ratio', 0) - sentiment_data.get('negative_ratio', 0)
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, ratios, color=['green', 'red', 'gray'])
    ax.set_title(f"Sentiment Distribution (Avg Rating: {sentiment_data.get('avg_rating', 0):.2f})")
    ax.set_ylabel("Ratio")
    st.pyplot(fig)
    plt.close(fig)


def plot_cluster_sizes(df):
    """
    Plot number of reviews per cluster
    """
    cluster_counts = df['cluster'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    cluster_counts.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title("Number of Reviews per Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)


def generate_wordcloud(keywords_text, title="Keywords WordCloud", container=None):
    """
    Generate a word cloud for top keywords.
    Renders into the given Streamlit container (e.g. a column), or the main page.
    """
    target = container or st
    if not keywords_text.strip():
        target.info(f"Skipping WordCloud for '{title}' — no keywords provided.")
        return

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(keywords_text)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=11, pad=8)
    fig.tight_layout()
    target.pyplot(fig)
    plt.close(fig)


def visualize_final_report(report_json_path):
    """
    Full visualizations from the final report JSON
    """
    with open(report_json_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    insight = report.get('insight', {})

    # Sentiment
    plot_sentiment(insight.get('sentiment', {}))

    # WordClouds in a single row
    dominant_keywords = " ".join(insight.get('dominant_theme', []))
    strengths_keywords = " ".join(insight.get('strengths', []))
    pain_points_keywords = " ".join(insight.get('pain_points', []))

    col1, col2, col3 = st.columns(3)
    generate_wordcloud(dominant_keywords, title="Dominant Theme", container=col1)
    generate_wordcloud(strengths_keywords, title="Strengths", container=col2)
    generate_wordcloud(pain_points_keywords, title="Pain Points", container=col3)