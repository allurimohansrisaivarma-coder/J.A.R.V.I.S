import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog

from jarvis.llm.base import Message

__all__ = ["Conversation", "ConversationManager"]

logger = structlog.get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are Jarvis, a desktop intelligence assistant. You are helpful, precise, and conversational.
You remember context from the current conversation and use it to provide relevant, concise answers.
When you don't know something, say so honestly rather than guessing.
Keep responses concise unless the user asks for detail."""


@dataclass
class Conversation:
    """Represents a single conversation session."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict = field(default_factory=dict)

    @property
    def user_messages(self) -> list[Message]:
        """Filter messages by user role."""
        return [m for m in self.messages if m.role == "user"]

    @property
    def assistant_messages(self) -> list[Message]:
        """Filter messages by assistant role."""
        return [m for m in self.messages if m.role == "assistant"]

    @property
    def token_estimate(self) -> int:
        """Rough token count estimate."""
        return sum(len(str(m.content)) for m in self.messages) // 4

    def add_message(self, message: Message) -> None:
        """Append a message and update timestamp."""
        self.messages.append(message)
        self.updated_at = datetime.now(UTC)

    def get_recent(self, n: int) -> list[Message]:
        """Get the last n messages."""
        return self.messages[-n:] if n > 0 else []

    def to_llm_messages(self, max_tokens: int = 8000) -> list[Message]:
        """Trim history to fit token budget, keeping system prompt + most recent messages."""
        if not self.messages:
            return []

        system_msgs = [m for m in self.messages if m.role == "system"]
        other_msgs = [m for m in self.messages if m.role != "system"]

        result = []
        result.extend(system_msgs)

        current_tokens = sum(len(str(m.content)) // 4 for m in system_msgs)

        recent: list[Message] = []
        for msg in reversed(other_msgs):
            msg_tokens = len(str(msg.content)) // 4
            if current_tokens + msg_tokens <= max_tokens:
                recent.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break

        result.extend(recent)
        return result


class ConversationManager:
    """Manages active conversation and history."""

    def __init__(self, system_prompt: str | None = None):
        """Initialize the conversation manager."""
        self._conversations: dict[str, Conversation] = {}
        self._active_id: str | None = None
        self._system_prompt: str = system_prompt or DEFAULT_SYSTEM_PROMPT
        logger.debug("ConversationManager initialized")

    def new_conversation(self) -> Conversation:
        """Create and activate a new conversation."""
        conv = Conversation()
        if self._system_prompt:
            conv.add_message(Message.system(self._system_prompt))
        self._conversations[conv.id] = conv
        self._active_id = conv.id
        logger.info("New conversation created", conv_id=conv.id)
        return conv

    def get_active(self) -> Conversation | None:
        """Get the active conversation."""
        if self._active_id:
            return self._conversations.get(self._active_id)
        return None

    def add_user_message(self, text: str) -> Message:
        """Add a user message to the active conversation."""
        conv = self.get_active()
        if not conv:
            conv = self.new_conversation()

        msg = Message.user(text)
        conv.add_message(msg)
        logger.debug("Added user message", conv_id=conv.id, length=len(text))
        return msg

    def add_assistant_message(self, text: str) -> Message:
        """Add an assistant message to the active conversation."""
        conv = self.get_active()
        if not conv:
            conv = self.new_conversation()

        msg = Message.assistant(text)
        conv.add_message(msg)
        logger.debug("Added assistant message", conv_id=conv.id, length=len(text))
        return msg

    def get_context_messages(self, max_tokens: int = 8000) -> list[Message]:
        """Get messages for LLM, including system prompt."""
        conv = self.get_active()
        if not conv:
            return []
        return conv.to_llm_messages(max_tokens=max_tokens)

    def end_conversation(self) -> None:
        """Mark current conversation as ended, clear active."""
        if self._active_id:
            logger.info("Ended conversation", conv_id=self._active_id)
            self._active_id = None

    def get_history(self) -> list[Conversation]:
        """Get all past conversations."""
        return list(self._conversations.values())
