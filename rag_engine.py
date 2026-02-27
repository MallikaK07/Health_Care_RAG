"""
Healthcare RAG Engine
Core module for document preprocessing, embedding, vector storage (ChromaDB),
and retrieval-augmented generation for clinical data analysis.
"""

import os
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ─── Configuration ───────────────────────────────────────────────────────────

DOCS_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# ─── Document Loading & Preprocessing ────────────────────────────────────────

def load_documents(docs_dir: str = DOCS_DIR) -> list[Document]:
    """Load all .txt files from the documents directory."""
    documents = []
    for filepath in sorted(glob.glob(os.path.join(docs_dir, "*.txt"))):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        doc = Document(
            page_content=text,
            metadata={"source": os.path.basename(filepath)},
        )
        documents.append(doc)
    print(f"✅ Loaded {len(documents)} document(s) from {docs_dir}")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunk(s) from {len(documents)} document(s)")
    return chunks


# ─── Embedding & Vector Store ────────────────────────────────────────────────

def get_embeddings() -> HuggingFaceEmbeddings:
    """Initialize the sentence-transformer embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(chunks: list[Document], persist_dir: str = CHROMA_DIR) -> Chroma:
    """Create (or overwrite) a ChromaDB vector store from document chunks."""
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print(f"✅ Vector store built with {len(chunks)} chunks → {persist_dir}")
    return vectorstore


def load_vector_store(persist_dir: str = CHROMA_DIR) -> Chroma:
    """Load an existing ChromaDB vector store from disk."""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )


# ─── Retrieval ───────────────────────────────────────────────────────────────

def retrieve(query: str, vectorstore: Chroma, k: int = 4) -> list[Document]:
    """Retrieve the top-k most relevant chunks using similarity search."""
    return vectorstore.similarity_search(query, k=k)


def hybrid_retrieve(query: str, vectorstore: Chroma, k: int = 4) -> list[Document]:
    """
    Hybrid retrieval: combine dense vector similarity with basic keyword
    matching to improve precision for complex medical queries.
    """
    # Dense retrieval – get more candidates, then re-rank
    candidates = vectorstore.similarity_search(query, k=k * 3)

    # Simple keyword boost: score each candidate by keyword overlap
    query_tokens = set(query.lower().split())
    scored = []
    for doc in candidates:
        content_tokens = set(doc.page_content.lower().split())
        keyword_overlap = len(query_tokens & content_tokens) / max(len(query_tokens), 1)
        scored.append((keyword_overlap, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:k]]


# ─── Answer Generation (simple extractive) ───────────────────────────────────

def generate_answer(query: str, context_docs: list[Document]) -> dict:
    """
    Generate an answer from retrieved context.
    Uses a lightweight extractive approach (no external LLM API required).
    For production, swap this with an LLM call (e.g., HuggingFace Inference
    or OpenAI) via LangChain's LLM chain.
    """
    # Build a combined context string
    context_parts = []
    citations = []
    for i, doc in enumerate(context_docs, 1):
        context_parts.append(doc.page_content)
        source = doc.metadata.get("source", "unknown")
        if source not in citations:
            citations.append(source)

    context = "\n\n---\n\n".join(context_parts)

    # Simple extractive answer: return the most relevant context with citations
    answer = (
        f"Based on the retrieved clinical documents, here is the relevant information:\n\n"
        f"{context}"
    )

    return {
        "query": query,
        "answer": answer,
        "citations": citations,
        "num_sources": len(citations),
        "num_chunks": len(context_docs),
    }


# ─── Pipeline: Ingest ────────────────────────────────────────────────────────

def ingest_documents(docs_dir: str = DOCS_DIR, persist_dir: str = CHROMA_DIR) -> Chroma:
    """End-to-end ingestion pipeline: load → chunk → embed → store."""
    docs = load_documents(docs_dir)
    chunks = chunk_documents(docs)
    vectorstore = build_vector_store(chunks, persist_dir)
    return vectorstore


# ─── Pipeline: Query ─────────────────────────────────────────────────────────

def query_rag(
    query: str,
    persist_dir: str = CHROMA_DIR,
    k: int = 4,
    hybrid: bool = True,
) -> dict:
    """End-to-end query pipeline: load store → retrieve → generate answer."""
    vectorstore = load_vector_store(persist_dir)
    if hybrid:
        docs = hybrid_retrieve(query, vectorstore, k=k)
    else:
        docs = retrieve(query, vectorstore, k=k)
    return generate_answer(query, docs)


# ─── CLI helper ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Healthcare RAG Engine — Ingesting documents...")
    print("=" * 60)
    ingest_documents()

    print("\n" + "=" * 60)
    print("Running sample query...")
    print("=" * 60)
    result = query_rag("What are the first-line treatments for type 2 diabetes?")
    print(f"\nQuery : {result['query']}")
    print(f"Sources: {result['citations']}")
    print(f"\n{result['answer'][:500]}...")
