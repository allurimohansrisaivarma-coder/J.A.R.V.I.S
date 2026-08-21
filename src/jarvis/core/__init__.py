"""Core components of the Jarvis assistant.

The session module pulls in optional desktop, database, and model dependencies.
Keep it lazy so importing the lightweight conversation types has no startup side
effects (and does not load Torch merely to collect a unit test).
"""

from typing import TYPE_CHECKING, Any

from jarvis.core.conversation import Conversation, ConversationManager

if TYPE_CHECKING:
    from jarvis.core.session import SessionManager

__all__ = ["Conversation", "ConversationManager", "SessionManager"]


def __getattr__(name: str) -> Any:
    if name == "SessionManager":
        from jarvis.core.session import SessionManager

        return SessionManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
