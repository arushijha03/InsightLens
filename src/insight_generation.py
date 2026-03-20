# src/insight_generation.py

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from collections import Counter
import numpy as np


# -------------------------------
# 1. Split Reviews
# -------------------------------
def extract_signals(reviews):
    positives, negatives = [], []

    for r in reviews:
        rating = r.get("rating", 3)

        if rating >= 4:
            positives.append(r["clean_text"])
        elif rating <= 2:
            negatives.append(r["clean_text"])

    return positives, negatives


# -------------------------------
# 2. Hybrid Term Extraction
# -------------------------------
def get_top_terms(texts, top_k=10):

    if not texts:
        return []

    CUSTOM_STOPWORDS = {
        "coffee", "product", "amazon", "order",
        "buy", "item", "use", "one", "get",
        "like", "would", "could", "also",
        "drink", "cup", "make", "going",
        "really", "much"
    }

    STOPWORDS = ENGLISH_STOP_WORDS.union(CUSTOM_STOPWORDS)

    # -------------------------------
    # TF-IDF Extraction
    # -------------------------------
    try:
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=STOPWORDS,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9
        )

        X = vectorizer.fit_transform(texts)

        if X.shape[1] > 0:
            features = vectorizer.get_feature_names_out()
            scores = X.mean(axis=0).A1

            top_idx = scores.argsort()[::-1][:top_k]
            terms = [features[i] for i in top_idx]

            if terms:
                return terms

    except:
        pass

    # -------------------------------
    # Fallback → frequency
    # -------------------------------
    words = " ".join(texts).split()

    filtered = [
        w for w in words
        if w not in STOPWORDS
        and len(w) > 3
        and w.isalpha()
    ]

    counter = Counter(filtered)

    return [w for w, _ in counter.most_common(top_k)]


# -------------------------------
# 3. Sentiment Summary
# -------------------------------
def sentiment_summary(reviews):
    ratings = [r.get("rating", 3) for r in reviews]

    return {
        "avg_rating": round(np.mean(ratings), 2),
        "positive_ratio": round(sum(r >= 4 for r in ratings) / len(ratings), 2),
        "negative_ratio": round(sum(r <= 2 for r in ratings) / len(ratings), 2),
    }


# -------------------------------
# 4. Recommendation Engine
# -------------------------------
def generate_recommendation(pain_points, strengths):
    recs = []

    if any(w in " ".join(pain_points) for w in ["bitter", "burnt", "aftertaste"]):
        recs.append("Improve taste profile and reduce bitterness in formulation.")

    if any(w in " ".join(pain_points) for w in ["stale", "expired"]):
        recs.append("Improve freshness control and inventory turnover.")

    if any(w in " ".join(pain_points) for w in ["packaging", "broken", "leak"]):
        recs.append("Improve packaging durability to prevent damage during shipping.")

    if any(w in " ".join(strengths) for w in ["value", "cheap", "affordable"]):
        recs.append("Maintain competitive pricing as it is a key strength.")

    if not recs:
        recs.append("Monitor customer feedback trends to identify improvement areas.")

    return " ".join(recs)


# -------------------------------
# 5. Main Insight Function
# -------------------------------
def generate_insight(top_reviews, cluster_info=None, cluster_keywords=None):

    positives, negatives = extract_signals(top_reviews)

    pos_terms = get_top_terms(positives)
    neg_terms = get_top_terms(negatives)

    # -------------------------------
    # Clean weak / generic words
    # -------------------------------
    GENERIC_WORDS = {
        "time", "day", "thing", "way", "lot",
        "dinner", "morning", "night", "coffee"
    }

    WEAK_WORDS = {"good", "bad", "nice", "okay", "fine"}

    pos_terms = [w for w in pos_terms if w not in GENERIC_WORDS and w not in WEAK_WORDS]
    neg_terms = [w for w in neg_terms if w not in GENERIC_WORDS and w not in WEAK_WORDS]

    # Remove overlap
    pos_terms = [w for w in pos_terms if w not in neg_terms]
    neg_terms = [w for w in neg_terms if w not in pos_terms]

    # Prefer meaningful phrases
    pos_terms = sorted(pos_terms, key=lambda x: len(x.split()), reverse=True)
    neg_terms = sorted(neg_terms, key=lambda x: len(x.split()), reverse=True)

    # Fallback safety
    if not pos_terms:
        pos_terms = ["quality", "smooth taste"]

    if not neg_terms:
        neg_terms = ["bitter taste", "quality issues"]

    sentiment = sentiment_summary(top_reviews)

    # -------------------------------
    # Clean dominant theme
    # -------------------------------
    GENERIC_THEME_WORDS = {
        "amazon", "order", "product", "item",
        "cup", "price", "time", "starbucks"
    }

    dominant_theme = [
        w.strip()
        for w in (cluster_keywords or [])
        if w.strip() not in GENERIC_THEME_WORDS and len(w.strip()) > 3
    ][:5]

    if not dominant_theme:
        dominant_theme = pos_terms[:2] + neg_terms[:2]

    # -------------------------------
    # Key Observation
    # -------------------------------
    if neg_terms and pos_terms:
        key_observation = (
            f"A key issue is {neg_terms[0]}, while satisfied users mention {pos_terms[0]}"
        )
    else:
        key_observation = "Customer feedback is mixed with no dominant signal."

    return {
        "dominant_theme": dominant_theme,
        "strengths": pos_terms[:5],
        "pain_points": neg_terms[:5],
        "sentiment": sentiment,
        "key_observation": key_observation,
        "business_recommendation": generate_recommendation(neg_terms, pos_terms),
    }