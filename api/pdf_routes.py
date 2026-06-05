"""
PDF router — upload, index, query, and manage PDF documents.

Endpoints:
  POST   /pdf/upload          → upload PDF, extract & embed chunks into vector DB
  POST   /pdf/query           → ask question against one or all PDFs
  POST   /pdf/voice-query     → upload audio question, get answer + TTS
  GET    /pdf/list            → list all indexed PDFs
  DELETE /pdf/{name}          → remove a PDF from the vector DB
"""

from __future__ import annotations
import base64
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from api.schemas import (
    PDFUploadResponse,
    PDFQueryRequest, PDFQueryResponse,
    PDFListResponse, PDFDeleteResponse,
)
from core.llm import answer_from_pdf
from vector_db import save_pdf_chunks, retrieve_pdf_context, delete_pdf_chunks, list_indexed_pdfs
from pdf import process_pdf
from voice import transcribe_audio, synthesize_speech, detect_language

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pdf", tags=["PDF Analyser"])


# ── Upload & Index ────────────────────────────

@router.post("/upload", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file. Its text is extracted, chunked, embedded,
    and stored in the PDF knowledge vector collection.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    pdf_name = file.filename

    try:
        _, chunks = process_pdf(pdf_bytes)
    except Exception as e:
        logger.error("PDF processing error: %s", e)
        raise HTTPException(status_code=422, detail=f"Failed to process PDF: {e}")

    if not chunks:
        raise HTTPException(status_code=422, detail="No extractable text found in the PDF.")

    try:
        save_pdf_chunks(chunks, pdf_name=pdf_name)
    except Exception as e:
        logger.error("Embedding error: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to index PDF: {e}")

    return PDFUploadResponse(
        pdf_name=pdf_name,
        total_chunks=len(chunks),
        message=f"Successfully indexed '{pdf_name}' with {len(chunks)} chunks.",
    )


# ── Text Query ────────────────────────────────

@router.post("/query", response_model=PDFQueryResponse)
async def query_pdf(request: PDFQueryRequest):
    """
    Ask a question. Retrieves relevant PDF chunks via RAG, then
    uses Gemini to answer strictly from document context.
    """
    # Retrieve relevant chunks
    chunks = retrieve_pdf_context(
        query=request.question,
        pdf_name=request.pdf_name,
    )

    # Generate answer
    try:
        answer = answer_from_pdf(
            user_question=request.question,
            pdf_chunks=chunks,
            pdf_name=request.pdf_name or "all documents",
        )
    except Exception as e:
        logger.error("LLM error: %s", e)
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # Optionally TTS
    audio_b64 = None
    lang = detect_language(answer)
    if request.return_audio:
        try:
            audio_bytes = synthesize_speech(answer, lang=lang)
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        except Exception as e:
            logger.warning("TTS failed: %s", e)

    return PDFQueryResponse(
        question=request.question,
        answer=answer,
        pdf_name=request.pdf_name,
        chunks_used=len(chunks),
        audio_base64=audio_b64,
    )


# ── Voice Query ───────────────────────────────

@router.post("/voice-query", response_model=PDFQueryResponse)
async def voice_query_pdf(
    audio: UploadFile = File(..., description="WAV audio file"),
    pdf_name: str = Form(default=None),
    return_audio: bool = Form(default=True),
):
    """
    Ask a PDF question by voice. Returns text answer + optional TTS audio.
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    try:
        question = transcribe_audio(audio_bytes)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if not question:
        raise HTTPException(status_code=400, detail="Could not transcribe audio.")

    query_req = PDFQueryRequest(
        question=question,
        pdf_name=pdf_name,
        return_audio=return_audio,
    )
    return await query_pdf(query_req)


# ── Management ────────────────────────────────

@router.get("/list", response_model=PDFListResponse)
async def list_pdfs():
    """List all PDFs currently indexed in the vector database."""
    pdfs = list_indexed_pdfs()
    return PDFListResponse(pdfs=pdfs, total=len(pdfs))


@router.delete("/{pdf_name}", response_model=PDFDeleteResponse)
async def delete_pdf(pdf_name: str):
    """Remove all chunks for a specific PDF from the vector database."""
    count = delete_pdf_chunks(pdf_name)
    if count == 0:
        raise HTTPException(status_code=404, detail=f"No chunks found for '{pdf_name}'.")
    return PDFDeleteResponse(
        pdf_name=pdf_name,
        chunks_deleted=count,
        message=f"Deleted {count} chunks for '{pdf_name}'.",
    )
