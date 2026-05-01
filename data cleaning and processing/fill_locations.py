import pandas as pd
import re
import random

# Load dataset
df = pd.read_csv("final_dataset_ready.csv")

# Adjectives
colors = ["black", "white", "blue", "brown", "grey", "red"]

def clean_text(text):
    text = str(text).lower()

    # Keep meaningful patterns like "64gb", "iphone 15"
    # Remove standalone numbers (time, random digits)
    text = re.sub(r'\b\d{1,2}\b', '', text)  # removes small useless numbers

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()

    # Remove duplicates
    seen = set()
    clean_words = []
    for w in words:
        if w not in seen:
            clean_words.append(w)
            seen.add(w)

    # If too short → add adjective
    if len(clean_words) == 1:
        adj = random.choice(colors)
        clean_words.insert(0, adj)

    return " ".join(clean_words[:5])

# Apply
df["clean_text"] = df["clean_text"].apply(clean_text)

# Save
df.to_csv("final_dataset_refined.csv", index=False)

print("✅ Saved as final_dataset_refined.csv")