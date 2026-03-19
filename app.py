import streamlit as st
import pandas as pd
from src.pipeline import full_pipeline

st.set_page_config(page_title="InsightLens Dashboard", layout="wide")
st.title("InsightLens: Amazon Review Insights")
st.markdown("Extract actionable insights from Amazon product reviews.")

st.sidebar.header("Query Settings")

query = st.sidebar.text_input("Enter your query:", value="Top complaints about coffee taste?")
top_k = st.sidebar.slider("Number of top reviews to retrieve:", min_value=1, max_value=20, value=5)
run_button = st.sidebar.button("Run Pipeline")

@st.cache_data(show_spinner=True)
def run_full_pipeline(query, k):
    return full_pipeline(query, k)

if run_button:
    with st.spinner("Running pipeline..."):
        output = run_full_pipeline(query, top_k)

if run_button:
    st.subheader("Top Reviews")
    for i, review in enumerate(output["top_reviews"], 1):
        st.markdown(f"**{i}.** {review}")

if run_button:
    st.subheader("Cluster Information")
    cluster_info = output["cluster_info"]
    st.markdown(f"**Dominant Cluster:** {cluster_info['dominant_cluster']}")
    st.markdown("**Top Clusters:**")
    st.table(pd.DataFrame(cluster_info["top_clusters"], columns=["Cluster ID", "Count"]))

if run_button:
    st.subheader("Structured Insight")
    insight = output["insight"]
    st.markdown(f"**Dominant Cluster:** {insight['dominant_cluster']}")
    st.markdown(f"**Keywords:** {insight['keywords']}")
    st.markdown(f"**Insight Text:** {insight['insight_text']}")

if run_button:
    st.subheader("Summary")
    summary = output["summary"]
    st.markdown("**Short Summary:**")
    st.write(summary["short_summary"])
    st.markdown("**Detailed Summary:**")
    st.write(summary["detailed_summary"])

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
        results[q] = run_full_pipeline(q, top_k)
    st.write("Batch results saved in memory for display or export.")

import json
if batch_button:
    with open("reports/batch_insights.json", "w") as f:
        json.dump(results, f, indent=2)
    st.success("Batch insights saved to reports/batch_insights.json")