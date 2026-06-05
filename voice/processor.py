"""
Voice processor — Speech-to-Text (Google Speech Recognition) + Text-to-Speech (gTTS).
"""

from __future__ import annotations
import io
import logging
from config import get_settings

logger = logging.getLogger(__name__)


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Convert WAV audio bytes to text using Google Speech Recognition.

    Args:
        audio_bytes: raw WAV audio data

    Returns:
        transcribed text string (empty string if nothing recognized)

    Raises:
        RuntimeError: if speech recognition service is unavailable
    """
    try:
        import speech_recognition as sr
    except ImportError:
        raise RuntimeError("speechrecognition package is not installed.")

    settings = get_settings()
    recognizer = sr.Recognizer()

    try:
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language=settings.voice_language)
        logger.info("Transcribed: %s", text[:100])
        return text
    except sr.UnknownValueError:
        logger.warning("Could not understand audio.")
        return ""
    except sr.RequestError as e:
        raise RuntimeError(f"Google Speech Recognition service error: {e}")
    except Exception as e:
        logger.error("Transcription error: %s", e)
        raise RuntimeError(f"Transcription failed: {e}")


def synthesize_speech(text: str, lang: str | None = None) -> bytes:
    """
    Convert text to speech using gTTS.

    Args:
        text: text to speak
        lang: BCP-47 language code (e.g. 'en', 'hi'). Defaults to settings.tts_language.

    Returns:
        MP3 audio bytes

    Raises:
        RuntimeError: if TTS fails
    """
    try:
        from gtts import gTTS
    except ImportError:
        raise RuntimeError("gtts package is not installed.")

    settings = get_settings()
    language = lang or settings.tts_language

    try:
        tts = gTTS(text=text, lang=language, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        logger.info("Synthesized speech: %d chars in lang=%s", len(text), language)
        return audio_buffer.read()
    except Exception as e:
        raise RuntimeError(f"TTS synthesis failed: {e}")


def detect_language(text: str) -> str:
    """
    Heuristic language detection based on Unicode character ranges.
    Returns a BCP-47 language code string.
    """
    if not text:
        return "en"

    # Check for Devanagari (Hindi)
    devanagari_chars = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    if devanagari_chars > len(text) * 0.1:
        return "hi"

    # Check for Arabic
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    if arabic_chars > len(text) * 0.1:
        return "ar"

    # Check for CJK (Chinese/Japanese/Korean)
    cjk_chars = sum(1 for c in text if "\u4E00" <= c <= "\u9FFF")
    if cjk_chars > len(text) * 0.1:
        return "zh"

    return "en"
