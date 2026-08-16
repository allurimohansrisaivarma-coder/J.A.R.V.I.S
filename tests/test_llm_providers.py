"""Tests for LLM provider base classes and data models."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from jarvis.llm.base import (
    AuthenticationError,
    LLMError,
    LLMProvider,
    LLMResponse,
    Message,
    ModelTier,
    ProviderUnavailableError,
    RateLimitError,
    TokenUsage,
    _TimingContext,
)


class TestMessage:
    """Tests for the Message dataclass."""

    def test_system_message(self):
        """System message factory creates correct role."""
        msg = Message.system("You are a helpful assistant.")
        assert msg.role == "system"
        assert msg.content == "You are a helpful assistant."
        assert msg.name is None

    def test_user_message(self):
        """User message factory creates correct role."""
        msg = Message.user("Hello, Jarvis!")
        assert msg.role == "user"
        assert msg.content == "Hello, Jarvis!"

    def test_assistant_message(self):
        """Assistant message factory creates correct role."""
        msg = Message.assistant("How can I help?")
        assert msg.role == "assistant"
        assert msg.content == "How can I help?"

    def test_user_message_with_list_content(self):
        """User message should accept list content for multimodal."""
        content = [{"type": "text", "text": "What's in this image?"}]
        msg = Message.user(content)
        assert msg.role == "user"
        assert isinstance(msg.content, list)

    def test_message_with_name(self):
        """Message should accept an optional name field."""
        msg = Message(role="user", content="hello", name="Alice")
        assert msg.name == "Alice"


class TestTokenUsage:
    """Tests for the TokenUsage dataclass."""

    def test_total_tokens(self):
        """Total tokens should be the sum of prompt and completion."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert usage.total_tokens == 150

    def test_default_values(self):
        """Default token counts should be zero."""
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_large_values(self):
        """Should handle large token counts."""
        usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=500_000)
        assert usage.total_tokens == 1_500_000


class TestLLMResponse:
    """Tests for the LLMResponse dataclass."""

    def test_response_creation(self):
        """LLMResponse should store all fields correctly."""
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20)
        resp = LLMResponse(
            content="Hello!",
            model="gemini-1.5-flash",
            provider="gemini",
            usage=usage,
            latency_ms=150.5,
        )
        assert resp.content == "Hello!"
        assert resp.model == "gemini-1.5-flash"
        assert resp.provider == "gemini"
        assert resp.usage.total_tokens == 30
        assert resp.latency_ms == 150.5
        assert resp.finish_reason == "stop"  # default
        assert resp.raw == {}  # default

    def test_response_custom_finish_reason(self):
        """LLMResponse should accept custom finish reasons."""
        resp = LLMResponse(
            content="...",
            model="test",
            provider="test",
            usage=TokenUsage(),
            latency_ms=0,
            finish_reason="length",
        )
        assert resp.finish_reason == "length"


class TestModelTier:
    """Tests for the ModelTier enum."""

    def test_tier_values(self):
        """Ensure all tier values are correct strings."""
        assert ModelTier.FAST.value == "fast"
        assert ModelTier.STANDARD.value == "standard"
        assert ModelTier.COMPLEX.value == "complex"

    def test_tier_count(self):
        """There should be exactly 3 tiers."""
        assert len(ModelTier) == 3


class TestTimingContext:
    """Tests for the _TimingContext manager."""

    def test_timing_measures_latency(self):
        """Timing context should measure non-zero latency."""
        ctx = _TimingContext()
        with ctx:
            # Simulate some work
            _ = sum(range(1000))
        assert ctx.latency_ms > 0

    def test_timing_latency_reasonable(self):
        """Measured latency should be reasonable (< 1 second for trivial work)."""
        ctx = _TimingContext()
        with ctx:
            pass
        assert ctx.latency_ms < 1000


class TestExceptionHierarchy:
    """Tests for the exception hierarchy."""

    def test_rate_limit_is_llm_error(self):
        """RateLimitError should be a subclass of LLMError."""
        assert issubclass(RateLimitError, LLMError)

    def test_auth_is_llm_error(self):
        """AuthenticationError should be a subclass of LLMError."""
        assert issubclass(AuthenticationError, LLMError)

    def test_unavailable_is_llm_error(self):
        """ProviderUnavailableError should be a subclass of LLMError."""
        assert issubclass(ProviderUnavailableError, LLMError)

    def test_catch_all_errors(self):
        """All provider errors should be catchable as LLMError."""
        with pytest.raises(LLMError):
            raise RateLimitError("rate limited")

        with pytest.raises(LLMError):
            raise AuthenticationError("bad key")

        with pytest.raises(LLMError):
            raise ProviderUnavailableError("down")


class TestGeminiProviderMessageConversion:
    """Test Gemini provider's message conversion logic."""

    def test_system_message_extraction(self):
        """System messages should be extracted as system_instruction."""
        from jarvis.llm.gemini import GeminiProvider

        provider = GeminiProvider.__new__(GeminiProvider)
        provider._model_name = "gemini-1.5-flash"

        messages = [
            Message.system("You are Jarvis."),
            Message.user("Hello"),
        ]
        contents, sys_prompt = provider._convert_messages(messages)

        assert sys_prompt == "You are Jarvis."
        assert len(contents) == 1  # Only the user message

    def test_role_mapping(self):
        """Assistant role should map to 'model' for Gemini."""
        from jarvis.llm.gemini import GeminiProvider

        provider = GeminiProvider.__new__(GeminiProvider)
        provider._model_name = "gemini-1.5-flash"

        messages = [
            Message.user("Hi"),
            Message.assistant("Hello!"),
            Message.user("How are you?"),
        ]
        contents, _ = provider._convert_messages(messages)

        assert len(contents) == 3
        assert contents[0].role == "user"
        assert contents[1].role == "model"
        assert contents[2].role == "user"


class TestGroqProviderMessageConversion:
    """Test Groq provider's message conversion logic."""

    def test_basic_conversion(self):
        """Messages should convert to OpenAI-compatible format."""
        from jarvis.llm.groq_provider import GroqProvider

        provider = GroqProvider.__new__(GroqProvider)

        messages = [
            Message.user("Hello"),
            Message.assistant("Hi there!"),
        ]
        formatted = provider._convert_messages(messages)

        assert len(formatted) == 2
        assert formatted[0] == {"role": "user", "content": "Hello"}
        assert formatted[1] == {"role": "assistant", "content": "Hi there!"}

    def test_system_prompt_prepended(self):
        """System prompt should be prepended as a system message."""
        from jarvis.llm.groq_provider import GroqProvider

        provider = GroqProvider.__new__(GroqProvider)

        messages = [Message.user("Hello")]
        formatted = provider._convert_messages(messages, system_prompt="Be helpful")

        assert len(formatted) == 2
        assert formatted[0] == {"role": "system", "content": "Be helpful"}
        assert formatted[1] == {"role": "user", "content": "Hello"}


@pytest.mark.integration
class TestRealAPIIntegration:
    """Integration tests that require real API keys. Skipped by default."""

    async def test_gemini_health_check(self):
        """Test that Gemini health check works with a real key."""
        import os
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            pytest.skip("GEMINI_API_KEY not set")

        from jarvis.llm.gemini import GeminiProvider
        provider = GeminiProvider(api_key=key)
        assert await provider.health_check() is True

    async def test_groq_health_check(self):
        """Test that Groq health check works with a real key."""
        import os
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            pytest.skip("GROQ_API_KEY not set")

        from jarvis.llm.groq_provider import GroqProvider
        provider = GroqProvider(api_key=key)
        assert await provider.health_check() is True
