"""
Session manager — maintains per-user conversation history in memory.
Each session holds a sliding window of turns (up to max_history).
"""

from __future__ import annotations
import uuid
from collections import defaultdict, deque
from config import get_settings
import logging

logger = logging.getLogger(__name__)

# {session_id: deque of {"role": ..., "content": ...}}
_sessions: dict[str, deque] = defaultdict(lambda: deque(maxlen=get_settings().max_chat_history))


def create_session() -> str:
    """Create a new session and return its ID."""
    session_id = str(uuid.uuid4())
    _sessions[session_id]  # initialise empty deque
    logger.info("Created session: %s", session_id)
    return session_id


def add_turn(session_id: str, role: str, content: str) -> None:
    """Append a message turn to a session. role: 'user' | 'assistant'"""
    _sessions[session_id].append({"role": role, "content": content})


def get_history(session_id: str) -> list[dict]:
    """Return the full history list for a session."""
    return list(_sessions.get(session_id, []))


def clear_session(session_id: str) -> None:
    """Clear all turns in a session."""
    if session_id in _sessions:
        _sessions[session_id].clear()
        logger.info("Cleared session: %s", session_id)


def delete_session(session_id: str) -> None:
    """Remove the session entirely."""
    _sessions.pop(session_id, None)
    logger.info("Deleted session: %s", session_id)


def list_sessions() -> list[str]:
    return list(_sessions.keys())
