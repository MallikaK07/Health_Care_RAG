# Healthcare RAG System 🏥

**Retrieval-Augmented Generation for Clinical Data Analysis**

A mini prototype demonstrating a RAG (Retrieval-Augmented Generation) pipeline for intelligent querying of clinical research papers and patient documentation. Built with LangChain, ChromaDB, Hugging Face sentence-transformers, and Streamlit.

---

## ✨ Features

- **Document Ingestion Pipeline** — Loads, cleans, and chunks healthcare documents into semantic embeddings
- **Vector Database (ChromaDB)** — Stores embeddings for fast similarity search
- **Hybrid Retrieval** — Combines dense vector similarity with keyword matching for improved precision
- **Streamlit Web App** — Interactive UI with real-time query processing and citation tracking
- **Sentence-Transformer Embeddings** — Uses `all-MiniLM-L6-v2` for lightweight, high-quality embeddings

## 📁 Project Structure

```
Health_Care_RAG/
├── app.py              # Streamlit web application
├── rag_engine.py       # Core RAG engine (preprocessing, embedding, retrieval)
├── requirements.txt    # Python dependencies
├── sample_docs/        # Sample clinical documents for demo
│   ├── alzheimers_research.txt
│   ├── cancer_immunotherapy.txt
│   ├── covid19_long_term.txt
│   ├── diabetes_management.txt
│   └── hypertension_treatment.txt
└── README.md
```

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Health_Care_RAG.git
cd Health_Care_RAG
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

### 5. Ingest & Query

1. Click **🔄 Ingest / Re-ingest Documents** in the sidebar to build the vector store
2. Type a clinical query (e.g., *"What are the first-line treatments for hypertension?"*)
3. Click **Search** to retrieve answers with citations

## 🛠️ Tech Stack

| Component          | Technology                                |
| ------------------ | ----------------------------------------- |
| Framework          | LangChain                                 |
| Embeddings         | Hugging Face `all-MiniLM-L6-v2`           |
| Vector Database    | ChromaDB                                  |
| Web Interface      | Streamlit                                 |
| Language           | Python 3.10+                              |

## 🔍 How It Works

```
Documents → Chunking → Embedding → ChromaDB
                                        ↓
User Query → Embedding → Hybrid Search → Context Retrieval → Answer Generation
```

1. **Ingestion**: Text documents are split into overlapping chunks (500 chars, 50 overlap) and embedded using sentence-transformers
2. **Storage**: Embeddings are persisted in ChromaDB for efficient retrieval
3. **Retrieval**: User queries are embedded and matched against stored vectors using hybrid search (dense similarity + keyword overlap)
4. **Generation**: Retrieved context is assembled with source citations to produce an answer

## 📝 Notes

- This is an **academic prototype** — the answer generation uses extractive context rather than an external LLM API
- To integrate a full generative LLM (e.g., OpenAI GPT, Llama), swap the `generate_answer()` function in `rag_engine.py` with a LangChain LLM chain
- Add your own `.txt` documents to `sample_docs/` and re-ingest to expand the knowledge base


