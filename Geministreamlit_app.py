# streamlit_app.py

import streamlit as st
from rag_retriever import recommend_laptops_rag

st.set_page_config(page_title=" Flipkart Laptop Recommender", layout="wide")
st.title(" Flipkart Laptop Recommendation System")

query = st.text_input("What type of laptop are you looking for?")

if query:
    parsed, results = recommend_laptops_rag(query)
    st.markdown(f" **Gemini Parsed Query:** _{parsed}_")
    st.subheader(" Top Laptop Matches")

    for _, row in results.iterrows():
        st.markdown(f"**{row['Product_name']}** — ₹{row['Prices']}")
        st.write(row["doc"])
        st.markdown("---")
