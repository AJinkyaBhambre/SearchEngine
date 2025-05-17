import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env
load_dotenv()

# Configure Gemini with your API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to parse user query using Gemini
def parse_query_gemini(user_query: str) -> str:
    prompt = f"""
    You are an assistant that extracts key laptop requirements from a user query.
    Analyze and summarize it with user intent, category, specs, and budget.

    Query: "{user_query}"
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")  
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error parsing query: {e}"

# Test block (remove this if importing in Streamlit or another script)
if __name__ == "__main__":
    query = "I'm looking for a budget gaming laptop under 60000 with SSD"
    parsed = parse_query_gemini(query)
    print(" Parsed Query:", parsed)

