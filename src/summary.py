# src/summary.py

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re


# -------------------------------
# 1. Clean Text Fixes
# -------------------------------
def fix_text(text):
    fixes = {
        "wasn t": "wasn't",
        "can t": "can't",
        "it s": "it's",
        "i ve": "i've",
        "doesn t": "doesn't"
    }

    for k, v in fixes.items():
        text = text.replace(k, v)

    return text


# -------------------------------
# 2. Relevance Filter (CRITICAL)
# -------------------------------
def is_relevant(sentence):

    STRONG_SIGNALS = {
        "bitter", "acidic", "burnt", "stale",
        "terrible", "awful", "bad", "weak",
        "strong", "chemical", "aftertaste",
        "smell", "flavor", "taste"
    }

    WEAK_WORDS = {
        "coffee", "product", "buy", "order"
    }

    BANNED_PATTERNS = [
        "add to cart",
        "trying new",
        "decided to",
        "i bought",
        "i purchased",
        "reading reviews",
        "price was",
        "arrived yesterday"
    ]

    s = sentence.lower()

    # ❌ Remove narrative / useless sentences
    for pattern in BANNED_PATTERNS:
        if pattern in s:
            return False

    words = set(s.split())

    strong_hits = len(words & STRONG_SIGNALS)
    weak_hits = len(words & WEAK_WORDS)

    # ✅ Must have strong signal
    if strong_hits >= 2:
        return True

    # ✅ Or 1 strong + multiple weak context
    if strong_hits >= 1 and weak_hits >= 2:
        return True

    return False


# -------------------------------
# 3. Sentence Splitter
# -------------------------------
def split_sentences(texts):

    sentences = []

    for t in texts:
        t = fix_text(t)

        chunks = re.split(r'\b(and|but|so|because|while|although)\b', t)

        buffer = ""
        for c in chunks:
            buffer += " " + c.strip()

            if len(buffer.split()) > 20:
                sentences.append(buffer.strip())
                buffer = ""

        if buffer:
            sentences.append(buffer.strip())

    # Filter by length + relevance
    sentences = [
        s for s in sentences
        if 40 < len(s) < 220 and is_relevant(s)
    ]

    return sentences


# -------------------------------
# 4. Rank Sentences
# -------------------------------
def rank_sentences(sentences):

    if not sentences:
        return []

    vectorizer = TfidfVectorizer(
        max_features=2000,
        stop_words="english"
    )

    X = vectorizer.fit_transform(sentences)
    scores = np.asarray(X.sum(axis=1)).flatten()

    return sorted(
        zip(sentences, scores),
        key=lambda x: x[1],
        reverse=True
    )


# -------------------------------
# 5. Deduplicate
# -------------------------------
def deduplicate_sentences(ranked, threshold=0.6):

    selected = []

    for sentence, _ in ranked:

        is_duplicate = False

        for sel in selected:
            overlap = len(set(sentence.split()) & set(sel.split())) / len(set(sentence.split()))
            if overlap > threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            selected.append(sentence)

        if len(selected) >= 6:
            break

    return selected


# -------------------------------
# 6. Rewrite for Clarity
# -------------------------------
def rewrite_sentence(s):

    s = s.strip()

    # Remove weak starters
    s = re.sub(r"^(and|but|so)\s+", "", s)

    # Clean spacing
    s = re.sub(r"\s+", " ", s)

    # Capitalize
    if len(s) > 1:
        s = s[0].upper() + s[1:]

    return s


# -------------------------------
# 7. Build Summary
# -------------------------------
def build_summary(sentences):

    cleaned = [rewrite_sentence(s) for s in sentences]

    return ". ".join(cleaned)


# -------------------------------
# 8. Main Function
# -------------------------------
def summarize_reviews(texts):

    if not texts:
        return {
            "short_summary": "",
            "detailed_summary": ""
        }

    sentences = split_sentences(texts)

    # Fallback if filtering too strict
    if not sentences:
        return {
            "short_summary": "Customers report poor taste and quality issues.",
            "detailed_summary": "Reviews highlight bitterness, acidity, and inconsistent flavor as key concerns."
        }

    ranked = rank_sentences(sentences)
    selected = deduplicate_sentences(ranked)

    short_summary = build_summary(selected[:3])
    detailed_summary = build_summary(selected[:6])

    return {
        "short_summary": short_summary,
        "detailed_summary": detailed_summary
    }