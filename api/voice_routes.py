"""
Voice router — standalone TTS endpoint and audio transcription.

Endpoints:
  POST /voice/tts          → text → MP3 audio bytes
  POST /voice/transcribe   → audio file → transcript text
"""

from __future__ import annotations
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import Response

from api.schemas import TTSRequest, TranscriptionResponse
from voice import synthesize_speech, transcribe_audio

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice", tags=["Voice"])


@router.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech. Returns raw MP3 audio bytes.
    The frontend can play this directly via the Audio Web API.
    """
    try:
        mp3_bytes = synthesize_speech(request.text, lang=request.lang)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return Response(
        content=mp3_bytes,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=speech.mp3"},
    )


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(audio: UploadFile = File(...)):
    """
    Upload a WAV audio file and get the transcript back.
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    try:
        transcript = transcribe_audio(audio_bytes)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return TranscriptionResponse(transcript=transcript)
