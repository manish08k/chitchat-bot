"""
Vector store — ChromaDB-backed RAG storage.

Two collections:
  • chat_memory    : stores Q&A pairs from conversation history
  • pdf_knowledge  : stores chunked text from uploaded PDFs

Each document is stored with its embedding, text content, and metadata.
"""

from __future__ import annotations
import uuid
import chromadb
from chromadb.config import Settings as ChromaSettings
from config import get_settings
from core.embeddings import embed_text, embed_batch
import logging
import os

logger = logging.getLogger(__name__)

_client: chromadb.PersistentClient | None = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        settings = get_settings()
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB client initialized at: %s", settings.chroma_persist_dir)
    return _client


def _get_collection(name: str) -> chromadb.Collection:
    return _get_client().get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


# ──────────────────────────────────────────────
# CHAT MEMORY
# ──────────────────────────────────────────────

def save_chat_interaction(user_message: str, bot_response: str) -> str:
    """Embed and store a Q&A pair in the chat memory collection."""
    settings = get_settings()
    col = _get_collection(settings.chroma_chat_collection)

    combined = f"User: {user_message}\nAssistant: {bot_response}"
    vector = embed_text(combined)
    doc_id = str(uuid.uuid4())

    col.add(
        ids=[doc_id],
        embeddings=[vector],
        documents=[combined],
        metadatas=[{
            "user_message": user_message[:500],
            "bot_response": bot_response[:1000],
            "type": "chat",
        }],
    )
    logger.debug("Saved chat interaction id=%s", doc_id)
    return doc_id


def retrieve_relevant_chat_history(query: str, top_k: int | None = None) -> list[dict]:
    """Retrieve the most semantically similar past interactions for a query."""
    settings = get_settings()
    k = top_k or settings.top_k_results
    col = _get_collection(settings.chroma_chat_collection)

    if col.count() == 0:
        return []

    vector = embed_text(query)
    results = col.query(
        query_embeddings=[vector],
        n_results=min(k, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    items = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        items.append({
            "text": doc,
            "metadata": meta,
            "relevance_score": round(1 - dist, 4),
        })
    return items


# ──────────────────────────────────────────────
# PDF KNOWLEDGE BASE
# ──────────────────────────────────────────────

def save_pdf_chunks(chunks: list[str], pdf_name: str) -> list[str]:
    """Embed and store a list of text chunks from a PDF document."""
    settings = get_settings()
    col = _get_collection(settings.chroma_pdf_collection)

    vectors = embed_batch(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"pdf_name": pdf_name, "chunk_index": i} for i, _ in enumerate(chunks)]

    col.add(
        ids=ids,
        embeddings=vectors,
        documents=chunks,
        metadatas=metadatas,
    )
    logger.info("Stored %d chunks from PDF '%s'", len(chunks), pdf_name)
    return ids


def retrieve_pdf_context(query: str, pdf_name: str | None = None, top_k: int | None = None) -> list[dict]:
    """Retrieve relevant PDF chunks for a query. Optionally filter by pdf_name."""
    settings = get_settings()
    k = top_k or settings.top_k_results
    col = _get_collection(settings.chroma_pdf_collection)

    if col.count() == 0:
        return []

    vector = embed_text(query)
    where = {"pdf_name": pdf_name} if pdf_name else None

    results = col.query(
        query_embeddings=[vector],
        n_results=min(k, col.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    items = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        items.append({
            "text": doc,
            "metadata": meta,
            "relevance_score": round(1 - dist, 4),
        })
    return items


def delete_pdf_chunks(pdf_name: str) -> int:
    """Delete all chunks belonging to a specific PDF. Returns count deleted."""
    settings = get_settings()
    col = _get_collection(settings.chroma_pdf_collection)
    existing = col.get(where={"pdf_name": pdf_name})
    ids = existing["ids"]
    if ids:
        col.delete(ids=ids)
        logger.info("Deleted %d chunks for PDF '%s'", len(ids), pdf_name)
    return len(ids)


def list_indexed_pdfs() -> list[str]:
    """Return unique PDF names currently stored in the vector DB."""
    settings = get_settings()
    col = _get_collection(settings.chroma_pdf_collection)
    if col.count() == 0:
        return []
    all_meta = col.get(include=["metadatas"])["metadatas"]
    return list({m["pdf_name"] for m in all_meta})
