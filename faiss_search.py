import pandas as pd
import numpy as np
import faiss

def build_faiss_index():
    print("🔄 Loading embedded data...")
    df = pd.read_pickle("funds_with_embeddings.pkl")

    print("📐 Preparing FAISS index...")
    embedding_matrix = np.vstack(df["embedding"].values).astype("float32")
    dimension = embedding_matrix.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)

    faiss.write_index(index, "funds_index.faiss")
    print("✅ FAISS index saved as funds_index.faiss!")

if __name__ == "__main__":
    build_faiss_index()