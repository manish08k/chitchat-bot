from pydantic import BaseModel, Field
from typing import Optional


# ── Chat ──────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    session_id: Optional[str] = None
    return_audio: bool = False  # if True, response includes base64 TTS audio


class ChatResponse(BaseModel):
    session_id: str
    user_message: str
    assistant_response: str
    audio_base64: Optional[str] = None   # MP3 encoded as base64
    memory_used: int = 0                  # how many RAG chunks were used
    language: str = "en"


# ── Voice ─────────────────────────────────────

class TranscriptionResponse(BaseModel):
    transcript: str
    session_id: Optional[str] = None


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    lang: Optional[str] = None


# ── PDF ───────────────────────────────────────

class PDFUploadResponse(BaseModel):
    pdf_name: str
    total_chunks: int
    message: str


class PDFQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    pdf_name: Optional[str] = None   # None = search all indexed PDFs
    return_audio: bool = False


class PDFQueryResponse(BaseModel):
    question: str
    answer: str
    pdf_name: Optional[str]
    chunks_used: int
    audio_base64: Optional[str] = None


class PDFListResponse(BaseModel):
    pdfs: list[str]
    total: int


class PDFDeleteResponse(BaseModel):
    pdf_name: str
    chunks_deleted: int
    message: str


# ── Session ───────────────────────────────────

class SessionResponse(BaseModel):
    session_id: str
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    components: dict
