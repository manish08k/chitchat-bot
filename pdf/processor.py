"""
PDF processor — extracts text from PDFs and splits into overlapping chunks
suitable for vector embedding and RAG retrieval.
"""

from __future__ import annotations
import io
import re
import PyPDF2
from config import get_settings
import logging

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF given its raw bytes."""
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    pages_text = []
    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
            pages_text.append(text)
        except Exception as e:
            logger.warning("Failed to extract page %d: %s", page_num, e)
    return "\n\n".join(pages_text)


def clean_text(text: str) -> str:
    """Normalize whitespace and remove junk characters."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F\u0900-\u097F]+', ' ', text)  # keep ASCII + Devanagari
    return text.strip()


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """
    Split text into overlapping chunks by word count.

    Args:
        text: cleaned document text
        chunk_size: words per chunk (default from settings)
        overlap: overlapping words between adjacent chunks (default from settings)

    Returns:
        list of non-empty text chunks
    """
    settings = get_settings()
    size = chunk_size or settings.chunk_size
    ovlp = overlap or settings.chunk_overlap

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(words):
            break
        start += size - ovlp  # slide forward with overlap

    return chunks


def process_pdf(pdf_bytes: bytes) -> tuple[str, list[str]]:
    """
    Full pipeline: bytes → raw text → clean text → chunks.

    Returns:
        (full_text, chunks)
    """
    raw = extract_text_from_pdf(pdf_bytes)
    cleaned = clean_text(raw)
    chunks = chunk_text(cleaned)
    logger.info("PDF processed: %d chars, %d chunks", len(cleaned), len(chunks))
    return cleaned, chunks
