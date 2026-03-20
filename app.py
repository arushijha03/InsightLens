# app.py

import streamlit as st
import pandas as pd
import json
from bs4 import BeautifulSoup
from src.pipeline import full_pipeline  # ensure pipeline.py is in the same folder or use absolute import

from src.visualization import visualize_final_report

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="InsightLens Dashboard", layout="wide")
st.title("InsightLens: Amazon Review Insights")
st.markdown("Extract actionable insights from Amazon product reviews.")

# -------------------------------
# Sidebar - Single Query
# -------------------------------
st.sidebar.header("Query Settings")
query = st.sidebar.text_input("Enter your query:", value="Top complaints about coffee taste?")
top_k = st.sidebar.slider("Number of top reviews to retrieve:", min_value=1, max_value=20, value=5)
run_button = st.sidebar.button("Run Pipeline")

# -------------------------------
# Helper Functions
# -------------------------------
def clean_html(text):
    """Remove HTML tags from review text"""
    return BeautifulSoup(str(text), "html.parser").get_text()

@st.cache_data(show_spinner=True)
def run_full_pipeline_cached(query, k):
    return full_pipeline(query, k)

# -------------------------------
# Run Pipeline for Single Query
# -------------------------------
if run_button:
    with st.spinner("Running pipeline..."):
        output = run_full_pipeline_cached(query, top_k)

    # ---------------------------
    # Display Top Reviews
    # ---------------------------
    seen = set()
    display_num = 0
    st.subheader("Top Reviews")
    for review in output["top_reviews"]:
        text = review["clean_text"]
        if text in seen:
            continue
        seen.add(text)
        display_num += 1
        st.markdown(f"**{display_num}.** {clean_html(review['review_text'])} (Rating: {review.get('rating', 'N/A')})")

    # ---------------------------
    # Cluster Info + Structured Insight side by side
    # ---------------------------
    st.divider()
    col_cluster, col_insight = st.columns(2)

    with col_cluster:
        st.subheader("Cluster Information")
        cluster_info = output["cluster_info"]
        st.markdown(f"**Dominant Cluster:** {cluster_info.get('dominant_cluster', 'N/A')}")
        st.markdown("**Top Clusters:**")
        st.table(pd.DataFrame(cluster_info.get("top_clusters", []), columns=["Cluster ID", "Count"]))

    with col_insight:
        st.subheader("Structured Insight")
        insight = output["insight"]
        st.markdown(f"**Dominant Theme:** {', '.join(insight.get('dominant_theme', []))}")
        st.markdown(f"**Strengths:** {', '.join(insight.get('strengths', []))}")
        st.markdown(f"**Pain Points:** {', '.join(insight.get('pain_points', []))}")
        st.markdown(f"**Key Observation:** {insight.get('key_observation', '')}")
        st.markdown(f"**Business Recommendation:** {insight.get('business_recommendation', '')}")

    # ---------------------------
    # Summary
    # ---------------------------
    st.divider()
    st.subheader("Summary")
    col_short, col_detail = st.columns(2)
    summary = output.get("summary", {})
    with col_short:
        st.markdown("**Short Summary:**")
        st.write(summary.get("short_summary", ""))
    with col_detail:
        st.markdown("**Detailed Summary:**")
        st.write(summary.get("detailed_summary", ""))

    # ---------------------------
    # Visualizations
    # ---------------------------
    st.divider()
    st.subheader("Visualizations")
    if st.checkbox("Show Visualizations", value=True):
        # Save output temporarily to JSON to reuse existing visualization functions
        temp_json_path = "temp_report.json"
        with open(temp_json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        visualize_final_report(temp_json_path)

# -------------------------------
# Sidebar - Batch Queries
# -------------------------------
st.sidebar.header("Batch Queries (Optional)")
batch_queries = st.sidebar.text_area(
    "Enter queries (one per line):",
    value="Top complaints about coffee taste?\nSummarize 1-star reviews for dog treats"
)
batch_button = st.sidebar.button("Run Batch Queries")

if batch_button:
    queries = batch_queries.strip().split("\n")
    results = {}
    for q in queries:
        results[q] = run_full_pipeline_cached(q, top_k)

    st.write("Batch results saved in memory for display or export.")

    # Save batch results
    with open("reports/batch_insights.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    st.success("Batch insights saved to reports/batch_insights.json")