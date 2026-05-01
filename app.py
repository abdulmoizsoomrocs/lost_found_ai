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
st.set_page_config(page_title="Lost & Found AI", layout="wide", page_icon="🔍")

# Dark theme CSS
st.markdown("""
<style>
    .main { background-color: #0f1419; color: #e1e8ed; }
    h1, h2, h3 { color: #1da1f2; }
    .stTextInput input { background-color: #192734; color: #e1e8ed; border: 1px solid #38444d; padding: 12px; border-radius: 8px; }
    .stTextInput input:focus { border-color: #1da1f2; box-shadow: 0 0 10px rgba(29, 161, 242, 0.3); }
    .card { background: #192734; border: 1px solid #38444d; border-radius: 12px; padding: 16px; margin: 12px 0; }
    .card:hover { border-color: #1da1f2; box-shadow: 0 0 15px rgba(29, 161, 242, 0.2); }
</style>
""", unsafe_allow_html=True)

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
        
        for _, row in results.head(10).iterrows():
            with st.container():
                st.markdown(f"""
                <div class="card">
                    <b style="color: #1da1f2; font-size: 1.1em;">📦 {row['clean_text'].title()}</b><br>
                    <small style="color: #aab8c2;">
                        📍 <b>Location:</b> {row['location'].upper()} | 
                        🏷️ <b>Status:</b> <span style="color: {'#e0245e' if row['status'] == 'lost' else '#17bf63'};"><b>{row['status'].upper()}</b></span><br>
                        👤 <b>Contact:</b> {row['name']} | 
                        📞 <b>Phone:</b> {row['contact']}
                    </small>
                </div>
                """, unsafe_allow_html=True)