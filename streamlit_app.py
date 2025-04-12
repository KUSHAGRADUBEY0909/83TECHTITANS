import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import difflib

st.set_page_config(page_title="Find My Fund", page_icon="🔍")

def load_data():
    return pd.read_pickle("funds_with_embeddings.pkl")

def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def load_faiss_index():
    return faiss.read_index("funds_index.faiss")

df = load_data()
index = load_faiss_index()
model = load_model()

st.title("Find My Fund – AI-Powered Lookup")
query = st.text_input("Enter your fund query (even with typos or full sentences):")

if query:
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vec, 100)

    results = df.iloc[indices[0]].copy()
    results["Semantic Score"] = distances[0]
    results["Fuzzy Score"] = results["schemeName"].apply(lambda name: difflib.SequenceMatcher(None, query.lower(), str(name).lower()).ratio())
    results["Final Score"] = 0.5 * (1 - results["Semantic Score"]) + 0.5 * results["Fuzzy Score"]
    results = results.sort_values("Final Score", ascending=False).reset_index(drop=True)

    st.subheader("Most Relevant Match")
    st.dataframe(results.iloc[:1][["schemeName", "amcName", "category", "Final Score"]])

    st.subheader(" Top 5 Alternatives")
    st.dataframe(results.iloc[1:6][["schemeName", "amcName", "category", "Final Score"]])

    st.subheader("📊 Top 10 Broader Matches")
    st.dataframe(results.iloc[6:16][["schemeName", "amcName", "category", "Final Score"]])

st.markdown("Made for Find My Fund Hackathon 🚀")
