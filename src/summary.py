# src/summary.py
from transformers import pipeline

# Fallback: use GPT2 or distilgpt2 for quick summarization
generator = pipeline("text-generation", model="distilgpt2")

def summarize_reviews(reviews):
    """
    Summarize top reviews: fallback text-generation method
    """
    text = " ".join(reviews[:10])
    prompt = f"Summarize the following reviews:\n{text}\nSummary:"
    summary = generator(prompt, max_length=150, do_sample=False)
    return {
        "short_summary": summary[0]["generated_text"],
        "detailed_summary": text[:500] + "..."
    }