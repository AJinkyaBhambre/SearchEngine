import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Load .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load data and FAISS index
df = pd.read_csv("laptop_docs.csv")
index = faiss.read_index("laptop_index.faiss")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Groq LLM setup
groq_llm = ChatGroq(model="llama3-70b-8192", api_key=GROQ_API_KEY)

# Parse user query using Groq
def parse_query_groq(user_query):
    prompt = f"""
    You are a smart assistant. Extract the key features, requirements, and preferences from the following laptop query: "{user_query}"
    Return a cleaned and condensed version of the query.
    """
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=prompt)
    ]
    try:
        response = groq_llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        return f" Error parsing query: {str(e)}"

# Recommend laptops
def recommend_laptops(query, top_k=5):
    parsed_query = parse_query_groq(query)
    query_embedding = model.encode([parsed_query])[0]
    distances, indices = index.search(np.array([query_embedding]), top_k)
    results = df.iloc[indices[0]].copy()
    return parsed_query, results
