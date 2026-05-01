import streamlit as st
import pandas as pd
from src.search import predict_category, rank_results

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lost & Found AI",
    layout="wide",
    page_icon="⬡",
    initial_sidebar_state="expanded",
)

# ─── Data & Constants ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/processed.csv")

LOCATIONS = [
    "ab1", "ab2", "ab3", "ab4", "ab5",
    "canteen", "cc", "library", "garden",
    "parking", "love garden", "mosque", "hostel",
]

LOCATION_LABELS = {
    "ab1": "Block AB1", "ab2": "Block AB2", "ab3": "Block AB3",
    "ab4": "Block AB4", "ab5": "Block AB5", "canteen": "Canteen",
    "cc": "Community Center", "library": "Library", "garden": "Garden",
    "parking": "Parking", "love garden": "Love Garden",
    "mosque": "Mosque", "hostel": "Hostel",
}

# ─── Utility Functions ───────────────────────────────────────────────────────
def extract_location(query: str) -> str | None:
    q = query.lower()
    for loc in LOCATIONS:
        if loc in q:
            return loc
    return None

def detect_status(query: str) -> str | None:
    q = query.lower()
    if "lost" in q:
        return "lost"
    if "found" in q:
        return "found"
    return None

def filter_results(df, category, status, location):
    results = df[df["category"] == category].copy()
    if status == "lost":
        results = results[results["status"] == "found"]
    elif status == "found":
        results = results[results["status"] == "lost"]
    if location:
        results = results[results["location"] == location]
    return results

# ─── CSS Injection ───────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background-color: #0b0f14;
    color: #c9d1d9;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #8b949e;
    font-size: 0.85rem;
    line-height: 1.7;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }

/* ── Header ── */
.lf-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 0.25rem;
}
.lf-icon {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
}
.lf-title {
    font-size: 1.6rem;
    font-weight: 600;
    color: #e6edf3;
    margin: 0;
    letter-spacing: -0.02em;
}
.lf-subtitle {
    font-size: 0.85rem;
    color: #8b949e;
    margin: 0;
}
.lf-divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 1.5rem 0;
}

/* ── Search bar wrapper ── */
.search-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.4rem;
}
[data-testid="stTextInput"] input {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.65rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stTextInput"] input:focus {
    border-color: #388bfd !important;
    box-shadow: 0 0 0 3px rgba(56, 139, 253, 0.12) !important;
    outline: none !important;
}
[data-testid="stTextInput"] input::placeholder { color: #484f58 !important; }

/* ── Analysis chips ── */
.analysis-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 1.2rem 0;
}
.chip {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    border: 1px solid;
}
.chip-blue  { background: rgba(56,139,253,0.1); border-color: rgba(56,139,253,0.35); color: #79b8ff; }
.chip-teal  { background: rgba(29,158,117,0.1); border-color: rgba(29,158,117,0.35); color: #56d364; }
.chip-amber { background: rgba(186,117,23,0.1); border-color: rgba(186,117,23,0.35); color: #e3b341; }
.chip-red   { background: rgba(226,75,74,0.1);  border-color: rgba(226,75,74,0.35);  color: #f85149; }
.chip-dot   { width: 7px; height: 7px; border-radius: 50%; background: currentColor; flex-shrink: 0; }

/* ── Section heading ── */
.section-heading {
    font-size: 0.75rem;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1.8rem 0 0.9rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-heading::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #21262d;
}

/* ── Result count badge ── */
.result-badge {
    display: inline-block;
    background: rgba(56,139,253,0.12);
    color: #79b8ff;
    border: 1px solid rgba(56,139,253,0.3);
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 2px 10px;
    margin-left: 8px;
    vertical-align: middle;
}

/* ── Result card ── */
.result-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 10px;
    transition: border-color 0.18s, transform 0.18s, box-shadow 0.18s;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    border-radius: 12px 0 0 12px;
}
.result-card.status-found::before { background: #56d364; }
.result-card.status-lost::before  { background: #f85149; }

.result-card:hover {
    border-color: #30363d;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.35);
}
.card-item-name {
    font-size: 1rem;
    font-weight: 600;
    color: #e6edf3;
    margin: 0 0 10px;
    font-family: 'DM Sans', sans-serif;
}
.card-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    margin-top: 4px;
}
.card-meta-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.8rem;
    color: #8b949e;
}
.card-meta-item strong {
    color: #c9d1d9;
    font-weight: 500;
}
.status-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 10px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.status-pill.found { background: rgba(86,211,100,0.15); color: #56d364; }
.status-pill.lost  { background: rgba(248,81,73,0.15);  color: #f85149; }

/* ── Empty & error states ── */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: #484f58;
}
.empty-state .icon { font-size: 2.5rem; margin-bottom: 1rem; }
.empty-state p { font-size: 0.9rem; margin: 0; }

/* ── Stagger animation ── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.animate-card {
    animation: fadeSlideIn 0.3s ease forwards;
}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1.5rem;">
        <p style="font-size:1rem; font-weight:600; color:#e6edf3; margin:0 0 0.4rem;">Lost & Found AI</p>
        <p style="font-size:0.75rem; color:#8b949e; margin:0; line-height:1.6;">
            AI-powered item recovery system
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**How it works**")
    st.markdown("""
    Type a natural language query — mention what the item is, where it might be,
    and whether it was lost or found. The AI will classify the item, detect the
    location and status, then surface the most relevant matches.
    """)

    st.markdown("---")
    st.markdown("**Example queries**")
    examples = [
        "lost black wallet in library",
        "found keys near the canteen",
        "lost phone in hostel ab3",
        "found glasses at the mosque",
    ]
    for ex in examples:
        if st.button(f"↗  {ex}", key=ex, use_container_width=True):
            st.session_state["query_input"] = ex

    st.markdown("---")
    st.markdown("""
    <p style="font-size:0.72rem; color:#484f58;">
    Powered by Logistic Regression · TF-IDF · Cosine Similarity
    </p>
    """, unsafe_allow_html=True)

# ─── Main Area ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="lf-header">
    <div class="lf-icon">⬡</div>
    <div>
        <p class="lf-title">Lost & Found AI</p>
        <p class="lf-subtitle">Describe what you're looking for — the AI handles the rest</p>
    </div>
</div>
<hr class="lf-divider">
""", unsafe_allow_html=True)

# Search input — picks up sidebar example clicks via session state
default_query = st.session_state.get("query_input", "")
st.markdown('<p class="search-label">Search</p>', unsafe_allow_html=True)
query = st.text_input(
    label="search_query",
    value=default_query,
    placeholder='e.g. "lost black wallet in library"',
    label_visibility="collapsed",
)

# ─── Processing ──────────────────────────────────────────────────────────────
if query.strip():
    df = load_data()

    with st.spinner(""):
        category = predict_category(query)
        location = extract_location(query)
        status   = detect_status(query)
        results  = filter_results(df, category, status, location)
        if not results.empty:
            results = rank_results(query, results)

    # ── Analysis section ────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Analysis</div>', unsafe_allow_html=True)

    chips_html = '<div class="analysis-grid">'
    chips_html += f'<span class="chip chip-blue"><span class="chip-dot"></span>Category: <strong>{category.title()}</strong></span>'

    if location:
        label = LOCATION_LABELS.get(location, location.title())
        chips_html += f'<span class="chip chip-teal"><span class="chip-dot"></span>Location: <strong>{label}</strong></span>'

    if status:
        opposite = "found" if status == "lost" else "lost"
        color_cls = "chip-amber"
        chips_html += (
            f'<span class="chip {color_cls}"><span class="chip-dot"></span>'
            f'Query: <strong>{status.title()}</strong> → showing <strong>{opposite.title()}</strong> items</span>'
        )
    else:
        chips_html += '<span class="chip chip-amber"><span class="chip-dot"></span>Status: <strong>All</strong></span>'

    chips_html += "</div>"
    st.markdown(chips_html, unsafe_allow_html=True)

    # ── Results section ──────────────────────────────────────────────────────
    count = min(10, len(results))
    badge = f'<span class="result-badge">{count} result{"s" if count != 1 else ""}</span>'
    st.markdown(f'<div class="section-heading">Results {badge}</div>', unsafe_allow_html=True)

    if results.empty:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">◎</div>
            <p>No matching items found for this query.<br>
               Try broadening the search or removing the location.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        top_results = results.head(10)
        for delay_idx, (_, row) in enumerate(top_results.iterrows()):
            item_status = row["status"]
            loc_label   = LOCATION_LABELS.get(row["location"], row["location"].title())
            pill_cls    = "found" if item_status == "found" else "lost"
            card_cls    = "status-found" if item_status == "found" else "status-lost"
            delay_ms    = delay_idx * 45

            st.markdown(f"""
            <div class="result-card {card_cls} animate-card" style="animation-delay:{delay_ms}ms">
                <p class="card-item-name">{row['clean_text'].title()}</p>
                <div class="card-meta">
                    <span class="card-meta-item">
                        ◎ <strong>{loc_label}</strong>
                    </span>
                    <span class="card-meta-item">
                        <span class="status-pill {pill_cls}">{item_status}</span>
                    </span>
                    <span class="card-meta-item">
                        ☉ <strong>{row['name']}</strong>
                    </span>
                    <span class="card-meta-item">
                        ⌁ <strong style="font-family:'DM Mono',monospace; font-size:0.78rem;">{row['contact']}</strong>
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    # Landing / empty state
    st.markdown("""
    <div class="empty-state" style="padding: 4rem 2rem;">
        <div class="icon" style="font-size:3rem; color:#21262d;">⬡</div>
        <p style="font-size:1rem; color:#484f58; margin-top:1rem;">
            Enter a search query above to get started.<br>
            <span style="font-size:0.82rem;">Try something like <em>"lost keys near canteen"</em></span>
        </p>
    </div>
    """, unsafe_allow_html=True)
