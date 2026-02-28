# Healthcare RAG System 🏥

**Retrieval-Augmented Generation for Clinical Data Analysis**

A prototype demonstrating a RAG (Retrieval-Augmented Generation) pipeline for intelligent querying of clinical research papers and patient documentation. Built with LangChain, ChromaDB, Hugging Face sentence-transformers, Groq LLM inference, and Streamlit.

---

## ✨ Features

- **AI-Powered Answers (Groq LLM)** — Generates synthesized, evidence-based clinical answers using open-source LLMs (Llama 3.3, Mixtral, Gemma 2) via Groq's free API
- **Document Ingestion Pipeline** — Loads, cleans, and chunks healthcare documents into semantic embeddings
- **Vector Database (ChromaDB)** — Stores embeddings for fast similarity search
- **Hybrid Retrieval** — Combines dense vector similarity with keyword matching for improved precision
- **Streamlit Web App** — Interactive UI with real-time query processing, citation tracking, and model selection
- **Sentence-Transformer Embeddings** — Uses `all-MiniLM-L6-v2` for lightweight, high-quality embeddings
- **Extractive Fallback** — Works without an API key using extractive context retrieval

## 📁 Project Structure

```
Health_Care_RAG/
├── app.py              # Streamlit web application
├── rag_engine.py       # Core RAG engine (preprocessing, embedding, retrieval, LLM generation)
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

### 4. Get a Groq API Key (free)

1. Sign up at [console.groq.com](https://console.groq.com)
2. Navigate to **API Keys** → **Create API Key**
3. Copy the key (starts with `gsk_...`)

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

### 6. Ingest & Query

1. Click **🔄 Ingest / Re-ingest Documents** in the sidebar to build the vector store
2. Paste your **Groq API Key** in the sidebar under **🤖 AI Model (Groq)**
3. Select a model and set the temperature
4. Type a clinical query (e.g., *"What are the first-line treatments for hypertension?"*)
5. Click **Search** to get AI-generated answers with citations

> **Note:** The app works without an API key too — it will return raw extracted document chunks instead of AI-synthesized answers.

## 🛠️ Tech Stack

| Component          | Technology                                |
| ------------------ | ----------------------------------------- |
| Framework          | LangChain                                 |
| LLM Inference      | Groq (Llama 3.3, Mixtral, Gemma 2)        |
| Embeddings         | Hugging Face `all-MiniLM-L6-v2`           |
| Vector Database    | ChromaDB                                  |
| Web Interface      | Streamlit                                 |
| Language           | Python 3.10+                              |

## 🤖 Supported Models

| Model              | ID                          | Best For                    |
| ------------------ | --------------------------- | --------------------------- |
| Llama 3.3 70B      | `llama-3.3-70b-versatile`   | Highest quality answers     |
| Llama 3 8B         | `llama3-8b-8192`            | Fast, lightweight responses |
| Mixtral 8x7B       | `mixtral-8x7b-32768`        | Long context handling       |
| Gemma 2 9B         | `gemma2-9b-it`              | Google's instruction-tuned  |

## 🔍 How It Works

```
Documents → Chunking → Embedding → ChromaDB
                                        ↓
User Query → Embedding → Hybrid Search → Context Retrieval → Groq LLM → AI Answer
```

1. **Ingestion**: Text documents are split into overlapping chunks (500 chars, 50 overlap) and embedded using sentence-transformers
2. **Storage**: Embeddings are persisted in ChromaDB for efficient retrieval
3. **Retrieval**: User queries are embedded and matched against stored vectors using hybrid search (dense similarity + keyword overlap)
4. **Generation**: Retrieved context is sent to a Groq-hosted LLM with a medical system prompt to produce an evidence-based, synthesized answer with source citations

## 📝 Notes

- Groq's free tier has rate limits (~30 requests/min) — sufficient for prototyping and demos
- Add your own `.txt` documents to `sample_docs/` and re-ingest to expand the knowledge base
- The medical system prompt ensures the LLM only answers from the provided context and does not hallucinate
