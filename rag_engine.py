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
from langchain_core.messages import HumanMessage, SystemMessage

# ─── Configuration ───────────────────────────────────────────────────────────

DOCS_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# ─── Document Loading & Preprocessing ────────────────────────────────

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


# ─── Answer Generation ────────────────────────────────────────────────────────

MEDICAL_SYSTEM_PROMPT = """You are an expert healthcare AI assistant. Your role is to provide
accurate, evidence-based answers to clinical questions using ONLY the provided context
from medical research papers and clinical documentation.

Rules:
- Base your answer strictly on the provided context.
- If the context does not contain enough information, clearly state that.
- Use medical terminology appropriately and explain complex terms.
- Structure your answer with clear headings and bullet points when helpful.
- Always indicate the level of evidence when possible.
- Do NOT fabricate information not present in the context."""


def _extract_citations(context_docs: list[Document]) -> tuple[str, list[str]]:
    """Build context string and citation list from retrieved documents."""
    context_parts, citations = [], []
    for doc in context_docs:
        context_parts.append(doc.page_content)
        source = doc.metadata.get("source", "unknown")
        if source not in citations:
            citations.append(source)
    return "\n\n---\n\n".join(context_parts), citations


def generate_answer(query: str, context_docs: list[Document]) -> dict:
    """Fallback extractive answer — returns raw retrieved chunks."""
    context, citations = _extract_citations(context_docs)
    answer = (
        "Based on the retrieved clinical documents, here is the relevant "
        f"information:\n\n{context}"
    )
    return {
        "query": query,
        "answer": answer,
        "citations": citations,
        "num_sources": len(citations),
        "num_chunks": len(context_docs),
        "model": "extractive",
    }


def generate_answer_llm(
    query: str,
    context_docs: list[Document],
    groq_api_key: str,
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.3,
) -> dict:
    """
    Generate an AI-synthesized answer using Groq-hosted open-source LLMs.
    Falls back to extractive mode on any API error.
    """
    from langchain_groq import ChatGroq  # lazy import to avoid load if unused

    context, citations = _extract_citations(context_docs)

    user_prompt = (
        f"Context from clinical documents:\n\n{context}\n\n"
        f"---\n\nQuestion: {query}\n\n"
        "Provide a comprehensive, evidence-based answer."
    )

    try:
        llm = ChatGroq(
            api_key=groq_api_key,
            model=model_name,
            temperature=temperature,
        )
        response = llm.invoke([
            SystemMessage(content=MEDICAL_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        answer = response.content
    except Exception as exc:
        answer = (
            f"⚠️ LLM call failed ({exc}). "
            "Falling back to extractive results:\n\n" + context
        )
        model_name = "extractive (error)"

    return {
        "query": query,
        "answer": answer,
        "citations": citations,
        "num_sources": len(citations),
        "num_chunks": len(context_docs),
        "model": model_name,
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
    groq_api_key: str | None = None,
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.3,
) -> dict:
    """End-to-end query pipeline: load store → retrieve → generate answer."""
    vectorstore = load_vector_store(persist_dir)
    if hybrid:
        docs = hybrid_retrieve(query, vectorstore, k=k)
    else:
        docs = retrieve(query, vectorstore, k=k)

    if groq_api_key:
        return generate_answer_llm(
            query, docs, groq_api_key, model_name, temperature
        )
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
