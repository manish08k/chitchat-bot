"""
main.py — FastAPI application entry point.

Run:
    python run.py
  OR:
    PYTHONPATH=. uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Logging setup ─────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Ensure data directory exists ──────────────
os.makedirs("./data/chroma_db", exist_ok=True)
os.makedirs("./data/uploads", exist_ok=True)

# ── Lifespan: warm up embedding model ────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Advanced RAG Chatbot API...")
    try:
        from core.embeddings import _load_model
        _load_model()
        logger.info("✅ Embedding model ready.")
    except Exception as e:
        logger.warning("Embedding model preload failed (will load on first request): %s", e)
    yield
    logger.info("🛑 Shutting down.")


# ── App ───────────────────────────────────────

app = FastAPI(
    title="Advanced RAG Chatbot API",
    description="""
## 🤖 Advanced AI Chatbot with RAG, Voice & PDF Analysis

### Features
- **Chat with Memory**: Gemini-powered chat with vector-based RAG memory (ChromaDB)
- **Voice I/O**: Upload audio → transcribe → respond → synthesize TTS audio
- **PDF Analyser**: Upload PDFs, ask questions grounded in document content
- **Streaming**: SSE streaming for real-time token-by-token responses
- **Embeddings**: sentence-transformers for semantic search and retrieval

### Quick Start
1. Set `GEMINI_API_KEY` in your `.env` file
2. `POST /chat/session` → get a session_id
3. `POST /chat/message` → chat with text
4. `POST /pdf/upload` → index a PDF
5. `POST /pdf/query` → ask questions about your PDF
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────
from api import chat_router, pdf_router, voice_router

app.include_router(chat_router)
app.include_router(pdf_router)
app.include_router(voice_router)


# ── Health & Root ─────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "Advanced RAG Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    from config import get_settings
    settings = get_settings()
    return {
        "status": "healthy",
        "gemini_model": settings.gemini_model,
        "embedding_model": settings.embedding_model,
        "chroma_dir": settings.chroma_persist_dir,
        "components": {
            "llm": "gemini",
            "embeddings": "sentence-transformers",
            "vector_db": "chromadb",
            "tts": "gtts",
            "stt": "google-speech-recognition",
        },
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check logs for details."},
    )