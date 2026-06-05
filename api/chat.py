"""
Chat router — handles text and voice chat with RAG memory.

Endpoints:
  POST /chat/session          → create new session
  POST /chat/message          → send text message, get response
  POST /chat/voice            → upload audio, get text + audio response
  GET  /chat/history/{id}     → get session history
  DELETE /chat/session/{id}   → clear/delete session
  GET  /chat/stream           → SSE streaming response
"""

from __future__ import annotations
import base64
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from api.schemas import (
    ChatRequest, ChatResponse,
    TranscriptionResponse, SessionResponse,
)
from core.llm import chat_with_context, stream_chat
from vector_db import save_chat_interaction, retrieve_relevant_chat_history
from voice import transcribe_audio, synthesize_speech, detect_language
from utils import create_session, add_turn, get_history, clear_session, delete_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


# ── Session Management ────────────────────────

@router.post("/session", response_model=SessionResponse)
async def new_session():
    """Create a new chat session."""
    sid = create_session()
    return SessionResponse(session_id=sid, message="Session created successfully.")


@router.delete("/session/{session_id}", response_model=SessionResponse)
async def end_session(session_id: str):
    """Delete a chat session and its history."""
    delete_session(session_id)
    return SessionResponse(session_id=session_id, message="Session deleted.")


@router.delete("/session/{session_id}/clear", response_model=SessionResponse)
async def clear_session_history(session_id: str):
    """Clear chat history without deleting session."""
    clear_session(session_id)
    return SessionResponse(session_id=session_id, message="Session cleared.")


@router.get("/history/{session_id}")
async def get_session_history(session_id: str):
    """Return conversation history for a session."""
    history = get_history(session_id)
    return {"session_id": session_id, "history": history, "turns": len(history)}


# ── Text Chat ─────────────────────────────────

@router.post("/message", response_model=ChatResponse)
async def chat_message(request: ChatRequest):
    """
    Send a text message. Returns assistant response.
    Optionally returns TTS audio as base64 MP3.
    """
    # Resolve or create session
    session_id = request.session_id or create_session()

    # Retrieve relevant past memory (RAG)
    memory_chunks = retrieve_relevant_chat_history(request.message)

    # Get session history
    history = get_history(session_id)

    # Generate response
    try:
        response_text = chat_with_context(
            user_message=request.message,
            session_history=history,
            relevant_memory=memory_chunks,
        )
    except Exception as e:
        logger.error("LLM error: %s", e)
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    # Persist turns
    add_turn(session_id, "user", request.message)
    add_turn(session_id, "assistant", response_text)

    # Save to vector DB for future memory
    save_chat_interaction(request.message, response_text)

    # Optionally synthesize speech
    audio_b64 = None
    lang = detect_language(response_text)
    if request.return_audio:
        try:
            audio_bytes = synthesize_speech(response_text, lang=lang)
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        except Exception as e:
            logger.warning("TTS failed: %s", e)

    return ChatResponse(
        session_id=session_id,
        user_message=request.message,
        assistant_response=response_text,
        audio_base64=audio_b64,
        memory_used=len(memory_chunks),
        language=lang,
    )


# ── Voice Chat ────────────────────────────────

@router.post("/voice", response_model=ChatResponse)
async def voice_chat(
    audio: UploadFile = File(..., description="WAV audio file"),
    session_id: str = Form(default=None),
    return_audio: bool = Form(default=True),
):
    """
    Upload audio → transcribe → chat → return text + optional TTS audio.
    Supports WAV format (use browser MediaRecorder or any audio recorder).
    """
    # Read audio bytes
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    # Transcribe
    try:
        transcript = transcribe_audio(audio_bytes)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if not transcript:
        return ChatResponse(
            session_id=session_id or create_session(),
            user_message="",
            assistant_response="Sorry, I couldn't understand the audio. Please try again.",
            memory_used=0,
            language="en",
        )

    # Pass to text chat
    chat_req = ChatRequest(
        message=transcript,
        session_id=session_id,
        return_audio=return_audio,
    )
    return await chat_message(chat_req)


# ── Streaming Chat ────────────────────────────

@router.get("/stream")
async def stream_chat_endpoint(message: str, session_id: str = None):
    """
    Server-Sent Events (SSE) streaming chat endpoint.
    Connect with EventSource in the browser.
    """
    sid = session_id or create_session()
    memory_chunks = retrieve_relevant_chat_history(message)
    history = get_history(sid)

    def event_generator():
        full_response = []
        try:
            for chunk in stream_chat(message, history, memory_chunks):
                full_response.append(chunk)
                yield f"data: {chunk}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
        finally:
            complete_response = "".join(full_response)
            if complete_response:
                add_turn(sid, "user", message)
                add_turn(sid, "assistant", complete_response)
                save_chat_interaction(message, complete_response)
            yield f"data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Session-Id": sid,
        },
    )
