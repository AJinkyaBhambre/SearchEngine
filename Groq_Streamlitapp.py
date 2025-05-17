import streamlit as st
from query_rag_groq_faiss import recommend_laptops

st.set_page_config(page_title="Flipkart Laptop Recommender", layout="wide")
st.title(" Flipkart Laptop Recommendation ")

query = st.text_input(" Enter your laptop requirements:")

if query:
    parsed, result_df = recommend_laptops(query)
    st.markdown(f"### Gemini/Groq Parsed Query:\n*_{parsed}_*")
    st.subheader(" Top Laptop Matches")

    for _, row in result_df.iterrows():
        st.markdown(f"**{row['Product_name']}** - ₹{row['Prices']}")
        st.write(row["doc"])
        st.markdown("---")
