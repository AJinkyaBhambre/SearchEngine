import pandas as pd
import numpy as np
import re
import faiss
from sentence_transformers import SentenceTransformer

# Load the dataset
df = pd.read_csv("flipkart_laptop_model.csv")

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Combine important columns into one searchable document
df["doc"] = df.apply(lambda row: f"{row['Product_name']} {row['CPU']} {row['RAM']} {row['Storage']} ₹{row['Prices']} {row['Description']}", axis=1)
df["doc"] = df["doc"].apply(clean_text)

# Create sentence embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
docs = df["doc"].tolist()
embeddings = model.encode(docs, show_progress_bar=True)

# Store embeddings in FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save index and cleaned data
faiss.write_index(index, "laptop_index.faiss")
df.to_csv("laptop_docs.csv", index=False)

print("✅ FAISS index and cleaned data saved.")
