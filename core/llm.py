"""
Gemini LLM client — wraps google-genai SDK.

Provides:
  • chat_with_context()    : RAG-augmented chat with memory
  • answer_from_pdf()      : Q&A grounded in PDF chunks
  • stream_chat()          : streaming generator variant
"""

from __future__ import annotations
from google import genai
from google.genai import types
from config import get_settings
import logging

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set in .env")
        _client = genai.Client(api_key=settings.gemini_api_key)
        logger.info("Gemini client initialized, model: %s", settings.gemini_model)
    return _client


def _build_rag_context(relevant_chunks: list[dict]) -> str:
    if not relevant_chunks:
        return ""
    lines = ["### Relevant Context (from memory/knowledge base):"]
    for i, chunk in enumerate(relevant_chunks, 1):
        lines.append(f"\n[{i}] (score={chunk['relevance_score']})\n{chunk['text']}")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# CHAT WITH RAG MEMORY
# ──────────────────────────────────────────────

CHAT_SYSTEM_PROMPT = """You are an advanced, helpful AI assistant with memory.
You have access to relevant past conversations and context below.
Use the context to give accurate, personalized, and coherent answers.
If the context is not relevant, answer from your own knowledge.
Be concise, clear, and friendly. Respond in the same language as the user."""


def chat_with_context(
    user_message: str,
    session_history: list[dict],
    relevant_memory: list[dict],
) -> str:
    client = _get_client()
    settings = get_settings()
    rag_context = _build_rag_context(relevant_memory)

    system_content = CHAT_SYSTEM_PROMPT
    if rag_context:
        system_content += f"\n\n{rag_context}"

    # Build contents list from session history
    contents = []
    for turn in session_history[-10:]:
        role = "user" if turn["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=turn["content"])]))

    contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system_content),
    )
    return response.text.strip()


# ──────────────────────────────────────────────
# PDF Q&A
# ──────────────────────────────────────────────

PDF_SYSTEM_PROMPT = """You are an expert document analyst AI.
You have been provided with relevant excerpts from a PDF document.
Answer the user's question strictly based on the provided document context.
If the answer is not found in the context, say: "I couldn't find that information in the document."
Be precise and cite which part of the document supports your answer when possible."""


def answer_from_pdf(
    user_question: str,
    pdf_chunks: list[dict],
    pdf_name: str,
) -> str:
    client = _get_client()
    settings = get_settings()

    if not pdf_chunks:
        return f"No relevant content found in '{pdf_name}' for your question."

    context_lines = [f"### Document: {pdf_name}\n### Relevant Excerpts:"]
    for i, chunk in enumerate(pdf_chunks, 1):
        context_lines.append(f"\n[Excerpt {i}] (relevance={chunk['relevance_score']})\n{chunk['text']}")
    context = "\n".join(context_lines)

    prompt = f"""{context}

### User Question:
{user_question}

### Answer:"""

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=PDF_SYSTEM_PROMPT),
    )
    return response.text.strip()


# ──────────────────────────────────────────────
# STREAMING CHAT
# ──────────────────────────────────────────────

def stream_chat(
    user_message: str,
    session_history: list[dict],
    relevant_memory: list[dict],
):
    client = _get_client()
    settings = get_settings()
    rag_context = _build_rag_context(relevant_memory)

    system_content = CHAT_SYSTEM_PROMPT
    if rag_context:
        system_content += f"\n\n{rag_context}"

    contents = []
    for turn in session_history[-10:]:
        role = "user" if turn["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=turn["content"])]))

    contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

    for chunk in client.models.generate_content_stream(
        model=settings.gemini_model,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system_content),
    ):
        if chunk.text:
            yield chunk.text