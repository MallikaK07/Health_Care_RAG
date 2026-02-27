"""
Healthcare RAG System — Streamlit Web Application
Interactive interface for querying clinical research papers and patient documentation
using Retrieval-Augmented Generation with ChromaDB and sentence-transformers.
"""

import os
import time
import streamlit as st
from rag_engine import (
    ingest_documents,
    query_rag,
    load_vector_store,
    DOCS_DIR,
    CHROMA_DIR,
)

# ─── Page Configuration ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Healthcare RAG System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0d9488, #0284c7, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: -8px;
        margin-bottom: 28px;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 4px;
    }

    .citation-chip {
        display: inline-block;
        background: #1e3a5f;
        color: #7dd3fc;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        margin: 3px 4px;
        border: 1px solid #2563eb44;
    }

    .result-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 24px;
        margin-top: 16px;
        line-height: 1.7;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }

    .stTextInput>div>div>input {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State ───────────────────────────────────────────────────────────

if "ingested" not in st.session_state:
    st.session_state.ingested = False
if "history" not in st.session_state:
    st.session_state.history = []

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    k_results = st.slider("Number of results (k)", min_value=1, max_value=10, value=4)
    use_hybrid = st.toggle("Hybrid search (dense + keyword)", value=True)

    st.markdown("---")
    st.markdown("## 📂 Document Store")

    # Count available documents
    doc_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".txt")] if os.path.isdir(DOCS_DIR) else []
    st.markdown(f"**{len(doc_files)}** document(s) available")
    for f in doc_files:
        st.markdown(f"- `{f}`")

    st.markdown("---")

    # Ingest button
    if st.button("🔄 Ingest / Re-ingest Documents", use_container_width=True):
        with st.spinner("Ingesting documents..."):
            ingest_documents()
            st.session_state.ingested = True
        st.success("✅ Documents ingested successfully!")

    # Check if vector store already exists
    if os.path.isdir(CHROMA_DIR) and not st.session_state.ingested:
        st.session_state.ingested = True

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#64748b; font-size:0.78rem;'>"
        "Healthcare RAG v1.0<br>LangChain · ChromaDB · HuggingFace"
        "</div>",
        unsafe_allow_html=True,
    )

# ─── Main Content ────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">🏥 Healthcare RAG System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    "Retrieval-Augmented Generation for Clinical Data Analysis — "
    "Powered by LangChain, ChromaDB, and Sentence-Transformers"
    "</p>",
    unsafe_allow_html=True,
)

# Metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        '<div class="metric-card"><div class="metric-value">{}</div>'
        '<div class="metric-label">Documents</div></div>'.format(len(doc_files)),
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        '<div class="metric-card"><div class="metric-value">{}</div>'
        '<div class="metric-label">Queries Run</div></div>'.format(len(st.session_state.history)),
        unsafe_allow_html=True,
    )
with col3:
    mode_label = "Hybrid" if use_hybrid else "Dense"
    st.markdown(
        '<div class="metric-card"><div class="metric-value">{}</div>'
        '<div class="metric-label">Search Mode</div></div>'.format(mode_label),
        unsafe_allow_html=True,
    )
with col4:
    status = "Ready" if st.session_state.ingested else "Not Ingested"
    st.markdown(
        '<div class="metric-card"><div class="metric-value">{}</div>'
        '<div class="metric-label">Status</div></div>'.format(status),
        unsafe_allow_html=True,
    )

st.markdown("")

# ─── Query Input ─────────────────────────────────────────────────────────────

query = st.text_input(
    "🔍 Enter your clinical query",
    placeholder="e.g. What are the first-line treatments for hypertension?",
)

col_a, col_b = st.columns([1, 5])
with col_a:
    search_clicked = st.button("Search", type="primary", use_container_width=True)

# ─── Run Query ───────────────────────────────────────────────────────────────

if search_clicked and query:
    if not st.session_state.ingested:
        st.warning("⚠️ Please ingest documents first using the sidebar button.")
    else:
        with st.spinner("🔎 Retrieving relevant documents..."):
            start = time.time()
            result = query_rag(query, k=k_results, hybrid=use_hybrid)
            elapsed = time.time() - start

        st.session_state.history.append(
            {"query": query, "time": round(elapsed, 2), "sources": result["citations"]}
        )

        # Response header
        st.markdown("### 📋 Results")
        info_cols = st.columns(3)
        info_cols[0].metric("⏱️ Response Time", f"{elapsed:.2f}s")
        info_cols[1].metric("📄 Chunks Retrieved", result["num_chunks"])
        info_cols[2].metric("📚 Source Documents", result["num_sources"])

        # Citations
        st.markdown("**Citations:**")
        chips_html = "".join(
            f'<span class="citation-chip">📄 {c}</span>' for c in result["citations"]
        )
        st.markdown(chips_html, unsafe_allow_html=True)

        # Answer
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown(result["answer"])
        st.markdown("</div>", unsafe_allow_html=True)

elif search_clicked and not query:
    st.warning("Please enter a query.")

# ─── Query History ───────────────────────────────────────────────────────────

if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🕘 Recent Queries")
    for i, h in enumerate(reversed(st.session_state.history[-5:]), 1):
        st.markdown(
            f"**{i}.** {h['query']}  \n"
            f"⏱️ {h['time']}s · 📚 {', '.join(h['sources'])}"
        )
