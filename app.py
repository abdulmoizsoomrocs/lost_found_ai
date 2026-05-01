import streamlit as st
import pandas as pd
from src.search import predict_category, rank_results

# Load dataset
df = pd.read_csv("data/processed.csv")

# Locations
locations = [
    "ab1","ab2","ab3","ab4","ab5",
    "canteen","cc","library","garden",
    "parking","love garden","mosque","hostel"
]

def extract_location(query):
    for loc in locations:
        if loc in query.lower():
            return loc
    return None

def detect_status(query):
    if "lost" in query.lower():
        return "lost"
    elif "found" in query.lower():
        return "found"
    return None


# 🎨 UI DESIGN
st.set_page_config(page_title="Lost & Found AI", layout="centered")

st.title("🔍 Lost & Found AI System")
st.write("Search for lost or found items using AI")

query = st.text_input("Enter your query (e.g., 'lost black wallet in library')")

if query:
    category = predict_category(query)
    location = extract_location(query)
    status = detect_status(query)

    st.subheader("📊 Analysis")

    st.write(f"📌 **Category:** {category}")
    
    if location:
        st.write(f"📍 **Location:** {location}")
    
    if status:
        st.write(f"🔁 **Showing opposite of:** {status}")

    # Filter data
    results = df[df["category"] == category]

    if status == "lost":
        results = results[results["status"] == "found"]
    elif status == "found":
        results = results[results["status"] == "lost"]

    if location:
        results = results[results["location"] == location]

    # Ranking
    if not results.empty:
        results = rank_results(query, results)

    # Display
    st.subheader("📋 Results")

    if results.empty:
        st.error("❌ No matching items found")
    else:
        st.success(f"Showing top {min(10, len(results))} results")
        st.dataframe(
            results[["clean_text", "location", "status", "name", "contact"]].head(10),
            use_container_width=True
        )