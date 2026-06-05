from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_chat_collection: str = "chat_memory"
    chroma_pdf_collection: str = "pdf_knowledge"
    max_chat_history: int = 20
    voice_language: str = "en"
    tts_language: str = "en"
    top_k_results: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 50

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()