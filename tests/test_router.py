"""Tests for the model router."""

import pytest
from unittest.mock import AsyncMock, MagicMock, PropertyMock

from jarvis.config.settings import RouterSettings
from jarvis.llm.base import (
    LLMError,
    LLMProvider,
    LLMResponse,
    Message,
    ModelTier,
    TokenUsage,
)
from jarvis.llm.router import ModelRouter


def _make_mock_provider(name: str, tier: ModelTier) -> LLMProvider:
    """Create a mock LLM provider."""
    provider = MagicMock(spec=LLMProvider)
    type(provider).name = PropertyMock(return_value=name)
    type(provider).tier = PropertyMock(return_value=tier)
    provider.generate = AsyncMock(
        return_value=LLMResponse(
            content=f"Response from {name}",
            model=f"{name}-model",
            provider=name,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
            latency_ms=100.0,
        )
    )
    provider.stream = AsyncMock()
    return provider


@pytest.fixture
def router_settings() -> RouterSettings:
    """Create test router settings."""
    return RouterSettings(
        complexity_threshold=0.8,
        fast_keywords=["hi", "hello", "thanks", "yes", "no", "ok"],
    )


@pytest.fixture
def full_router(router_settings: RouterSettings) -> ModelRouter:
    """Create a router with all three tiers configured."""
    providers = {
        ModelTier.FAST: _make_mock_provider("groq", ModelTier.FAST),
        ModelTier.STANDARD: _make_mock_provider("gemini-flash", ModelTier.STANDARD),
        ModelTier.COMPLEX: _make_mock_provider("gemini-pro", ModelTier.COMPLEX),
    }
    return ModelRouter(providers=providers, settings=router_settings)


class TestIntentClassification:
    """Test the router's intent classification logic."""

    @pytest.mark.asyncio
    async def test_fast_keyword_routes_to_fast(self, full_router: ModelRouter):
        """Messages matching fast keywords should route to FAST tier."""
        messages = [Message.user("hi")]
        tier = await full_router._classify_intent(messages)
        assert tier == ModelTier.FAST

    @pytest.mark.asyncio
    async def test_simple_response_routes_to_fast(self, full_router: ModelRouter):
        """Simple conversational responses should route to FAST."""
        messages = [Message.user("yes")]
        tier = await full_router._classify_intent(messages)
        assert tier == ModelTier.FAST

    @pytest.mark.asyncio
    async def test_normal_question_routes_to_standard(self, full_router: ModelRouter):
        """Regular questions should route to STANDARD tier."""
        messages = [Message.user("How do I configure a Python virtual environment?")]
        tier = await full_router._classify_intent(messages)
        assert tier == ModelTier.STANDARD

    @pytest.mark.asyncio
    async def test_complex_keyword_routes_to_complex(self, full_router: ModelRouter):
        """Messages with complexity keywords should route to COMPLEX."""
        messages = [Message.user("Analyze the performance characteristics of this algorithm")]
        tier = await full_router._classify_intent(messages)
        assert tier == ModelTier.COMPLEX

    @pytest.mark.asyncio
    async def test_empty_messages_routes_to_standard(self, full_router: ModelRouter):
        """Empty message list should default to STANDARD."""
        tier = await full_router._classify_intent([])
        assert tier == ModelTier.STANDARD

    @pytest.mark.asyncio
    async def test_long_history_routes_to_complex(self, full_router: ModelRouter):
        """Very long conversation history should route to COMPLEX."""
        messages = [Message.user("x" * 5000) for _ in range(5)]
        tier = await full_router._classify_intent(messages)
        assert tier == ModelTier.COMPLEX


class TestRouting:
    """Test the router's routing and fallback logic."""

    @pytest.mark.asyncio
    async def test_route_calls_correct_provider(self, full_router: ModelRouter):
        """Route should call the provider matching the classified tier."""
        messages = [Message.user("hello")]
        response = await full_router.route(messages)
        assert response.provider == "groq"

    @pytest.mark.asyncio
    async def test_standard_route(self, full_router: ModelRouter):
        """Standard queries should route to gemini-flash."""
        messages = [Message.user("How do I create a class in Python?")]
        response = await full_router.route(messages)
        assert response.provider == "gemini-flash"

    @pytest.mark.asyncio
    async def test_fallback_on_provider_error(self, full_router: ModelRouter):
        """Fallback should try the next tier when primary fails."""
        messages = [Message.user("hello")]
        # Make FAST provider fail
        full_router.providers[ModelTier.FAST].generate = AsyncMock(
            side_effect=LLMError("Provider down")
        )
        # generate_with_fallback should fall back to STANDARD
        response = await full_router.generate_with_fallback(messages)
        assert response.provider == "gemini-flash"

    @pytest.mark.asyncio
    async def test_all_providers_fail_raises(self, full_router: ModelRouter):
        """If all providers fail, should raise LLMError."""
        messages = [Message.user("hello")]
        for provider in full_router.providers.values():
            provider.generate = AsyncMock(side_effect=LLMError("Down"))

        with pytest.raises(LLMError, match="All providers failed"):
            await full_router.generate_with_fallback(messages)


class TestProviderHealth:
    """Test provider health tracking."""

    def test_provider_starts_healthy(self, full_router: ModelRouter):
        """All providers should start as healthy."""
        for tier in ModelTier:
            assert full_router._is_provider_healthy(tier) is True

    def test_provider_unhealthy_after_failures(self, full_router: ModelRouter):
        """Provider should be marked unhealthy after 3 failures."""
        for _ in range(3):
            full_router._record_failure(ModelTier.FAST)
        assert full_router._is_provider_healthy(ModelTier.FAST) is False

    def test_two_failures_still_healthy(self, full_router: ModelRouter):
        """Provider should remain healthy with fewer than 3 failures."""
        full_router._record_failure(ModelTier.FAST)
        full_router._record_failure(ModelTier.FAST)
        assert full_router._is_provider_healthy(ModelTier.FAST) is True
