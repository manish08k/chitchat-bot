from .chat import router as chat_router
from .pdf_routes import router as pdf_router
from .voice_routes import router as voice_router

__all__ = ["chat_router", "pdf_router", "voice_router"]
