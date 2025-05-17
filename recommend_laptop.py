from query_parser import parse_query_gemini

query = "I'm looking for a budget gaming laptop under 60000 with SSD"
parsed = parse_query_gemini(query)
print("Parsed Query:", parsed)

