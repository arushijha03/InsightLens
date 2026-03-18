import re
from collections import Counter

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def extract_top_terms(text, top_k=10):
    stopwords = set([
        "the", "and", "is", "in", "it", "of", "to", "a", "this",
        "that", "was", "for", "with", "as", "but", "on", "are"
    ])
    words = [w for w in text.split() if w not in stopwords and len(w) > 2]
    freq = Counter(words)
    return [word for word, _ in freq.most_common(top_k)]

def detect_sentiment_patterns(reviews):
    positive_words = ["good", "great", "excellent", "love", "amazing", "best"]
    negative_words = ["bad", "worst", "awful", "terrible", "disappointing", "poor"]

    pos_count, neg_count = 0, 0
    for review in reviews:
        r = review.lower()
        if any(w in r for w in positive_words):
            pos_count += 1
        if any(w in r for w in negative_words):
            neg_count += 1
    return pos_count, neg_count

def generate_insight(reviews, cluster_info, cluster_keywords_df):
    dominant_cluster = cluster_info["dominant_cluster"]
    keywords_row = cluster_keywords_df[cluster_keywords_df["cluster"] == dominant_cluster]
    keywords = keywords_row["keywords"].values[0] if not keywords_row.empty else "N/A"

    combined_text = " ".join(reviews)
    cleaned_text = clean_text(combined_text)
    top_terms = extract_top_terms(cleaned_text)
    pos_count, neg_count = detect_sentiment_patterns(reviews)

    insight = f"Users frequently discuss topics related to {keywords}. Common terms include {', '.join(top_terms[:5])}. "

    if neg_count > pos_count:
        insight += "There is a noticeable trend of dissatisfaction, with users highlighting issues in quality, taste, or usability."
    elif pos_count > neg_count:
        insight += "Overall sentiment is positive, with users appreciating key aspects of the product experience."
    else:
        insight += "User sentiment appears mixed, with both positive and negative experiences reported."

    return {
        "dominant_theme": f"Cluster {dominant_cluster}",
        "keywords": keywords,
        "top_terms": top_terms[:10],
        "sentiment": {"positive_signals": pos_count, "negative_signals": neg_count},
        "insight": insight,
        "supporting_reviews": reviews[:5]
    }

def summarize_reviews(reviews):
    extractive_summary = reviews[:3]
    combined_text = " ".join(reviews[:10])
    short_summary = "Users share consistent feedback highlighting key product experience themes."
    detailed_summary = combined_text

    return {
        "extractive": extractive_summary,
        "short_summary": short_summary,
        "detailed_summary": detailed_summary
    }