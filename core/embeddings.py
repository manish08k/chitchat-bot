"""
Embedding engine — generates dense vector embeddings for text using
sentence-transformers. Singleton pattern; model loads once at startup.
"""

from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer
from config import get_settings
import logging

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def _load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        settings = get_settings()
        logger.info("Loading embedding model: %s", settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model)
        logger.info("Embedding model loaded.")
    return _model


def embed_text(text: str) -> list[float]:
    """Return a flat list of floats (embedding vector) for a single string."""
    model = _load_model()
    vector: np.ndarray = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Batch embed a list of strings. More efficient than calling embed_text() in a loop."""
    model = _load_model()
    vectors: np.ndarray = model.encode(texts, normalize_embeddings=True, batch_size=32)
    return vectors.tolist()
