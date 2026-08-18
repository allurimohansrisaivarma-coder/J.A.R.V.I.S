"""Abstract base classes and data models for LLM providers."""

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelTier(Enum):
    """Tier of the model to be used."""
    FAST = "fast"
    STANDARD = "standard"
    COMPLEX = "complex"


@dataclass
class Message:
    """A single message in a conversation."""
    role: str
    content: str | list
    name: str | None = None

    @classmethod
    def system(cls, text: str) -> "Message":
        """Create a system message."""
        return cls(role="system", content=text)

    @classmethod
    def user(cls, text: str | list) -> "Message":
        """Create a user message."""
        return cls(role="user", content=text)

    @classmethod
    def assistant(cls, text: str) -> "Message":
        """Create an assistant message."""
        return cls(role="assistant", content=text)

    @classmethod
    def tool(cls, content: str | list, name: str) -> "Message":
        """Create a tool response message."""
        return cls(role="tool", content=content, name=name)


@dataclass
class TokenUsage:
    """Token usage for an LLM request."""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens used."""
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResponse:
    """Standardized response from an LLM provider."""
    content: str
    model: str
    provider: str
    usage: TokenUsage
    latency_ms: float
    finish_reason: str = "stop"
    raw: dict | None = None
    tool_calls: list[Any] = field(default_factory=list)


class LLMError(Exception):
    """Base exception for LLM errors."""


class RateLimitError(LLMError):
    """Rate limit exceeded."""


class AuthenticationError(LLMError):
    """Authentication failed."""


class ProviderUnavailableError(LLMError):
    """Provider is currently unavailable."""


class _TimingContext:
    """Context manager to measure latency of an operation."""

    def __init__(self):
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()

    @property
    def latency_ms(self) -> float:
        """Get the latency in milliseconds."""
        return (self.end_time - self.start_time) * 1000


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the provider."""

    @property
    @abstractmethod
    def tier(self) -> ModelTier:
        """Model tier of the provider."""

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        tools: Any | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM."""

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        tools: Any | None = None,
    ) -> AsyncIterator[str]:
        """Stream a response from the LLM."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is healthy."""

    def _build_timing_context(self) -> _TimingContext:
        """Build a timing context to measure latency."""
        return _TimingContext()
