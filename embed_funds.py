import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

def generate_embeddings():
    print("🧠 Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("🔄 Loading cleaned data...")
    df = pd.read_pickle("clean_funds.pkl")

    print("🔍 Creating embedding text...")
    df["combined_text"] = df["schemeName"] + " | " + df["amcName"] + " | " + df["category"] + " > " + df["subCategory"] + " | " + df["objective"]

    print("⚡ Generating embeddings...")
    embeddings = model.encode(df["combined_text"].tolist(), batch_size=32, convert_to_numpy=True)

    df["embedding"] = list(embeddings)

    df.to_pickle("funds_with_embeddings.pkl")
    print("✅ Embeddings saved as funds_with_embeddings.pkl!")

if __name__ == "__main__":
    generate_embeddings()