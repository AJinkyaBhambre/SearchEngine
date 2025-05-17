import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from query_parser import parse_query_gemini  # You must have query_parser.py ready

# -------------------- Load environment and data --------------------
load_dotenv()

# Load the FAISS index and product data
df = pd.read_csv("laptop_docs.csv")
index = faiss.read_index("laptop_index.faiss")

# Load sentence embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# -------------------- RAG-style laptop retrieval --------------------
def recommend_laptops_rag(user_query: str, top_k: int = 5):
    # Step 1: Parse query with Gemini
    parsed_query = parse_query_gemini(user_query)

    # Step 2: Get embedding for parsed query
    query_embedding = embedding_model.encode([parsed_query])[0]

    # Step 3: Search FAISS index for similar embeddings
    distances, indices = index.search(np.array([query_embedding]), top_k)

    # Step 4: Retrieve matched laptops from dataframe
    matched_laptops = df.iloc[indices[0]].copy()

    return parsed_query, matched_laptops


if __name__ == "__main__":
    query = "I need a lightweight laptop for programming under 60000"
    parsed, results = recommend_laptops_rag(query)

    print("\n Parsed Query:", parsed)
    print("\n Top Laptop Matches:\n")
    for idx, row in results.iterrows():
        print(f"➡ {row['Product_name']} | ₹{row['Prices']}\n   {row['doc']}\n")




