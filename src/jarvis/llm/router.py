"""Model routing layer."""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import structlog

from jarvis.config.settings import RouterSettings
from jarvis.llm.base import (
    AuthenticationError,
    LLMError,
    LLMProvider,
    LLMResponse,
    Message,
    ModelTier,
    ProviderUnavailableError,
    RateLimitError,
    has_images,
)
from jarvis.utils.metrics import metrics

logger = structlog.get_logger(__name__)


class ModelRouter:
    """Routes LLM requests to appropriate models based on intent and load."""

    def __init__(
        self,
        providers: dict[ModelTier, LLMProvider],
        settings: RouterSettings,
        vision_fallback: LLMProvider | None = None,
    ):
        """Initialize the router."""
        self.providers = providers
        self.settings = settings
        self.vision_fallback = vision_fallback

        # Track provider failures (dict of tier to list of failure timestamps)
        self._provider_failures: dict[ModelTier, list[float]] = {tier: [] for tier in ModelTier}
        self._provider_ttft: dict[ModelTier, list[float]] = {tier: [] for tier in ModelTier}

    async def _classify_intent(self, messages: list[Message]) -> ModelTier:
        """Classify the complexity of the query to select the optimal model tier."""
        if not messages:
            return ModelTier.STANDARD

        has_image = False
        for m in messages:
            if isinstance(m.content, list):
                has_image = True
                break

        last_msg = str(messages[-1].content).lower()

        if has_image:
            logger.debug("router.classification", tier="COMPLEX", reason="image_input")
            return ModelTier.COMPLEX

        last_msg = str(messages[-1].content).lower()
        word_count = len(last_msg.split())

        # Check for fast keywords
        for keyword in self.settings.fast_keywords:
            if (
                keyword.lower() in last_msg.split()
                and word_count < 20
                and not any(
                    tech_term in last_msg
                    for tech_term in ["code", "function", "class", "def ", "class "]
                )
            ):
                logger.debug("router.classification", tier="FAST", reason="fast_keyword")
                return ModelTier.FAST

        # Simple conversational signals
        fast_signals = {"yes", "no", "ok", "thanks", "go ahead", "sure", "yep", "nope"}
        if last_msg.strip() in fast_signals:
            logger.debug("router.classification", tier="FAST", reason="simple_response")
            return ModelTier.FAST

        # Check for complex intent
        complex_keywords = [
            "analyze in detail",
            "deep analysis",
            "explain in detail",
            "in-depth analysis",
            "reason step by step",
            "design the architecture",
            "system architecture",
            "comprehensive analysis",
        ]
        if any(keyword in last_msg for keyword in complex_keywords):
            logger.debug("router.classification", tier="COMPLEX", reason="complex_keyword")
            return ModelTier.COMPLEX

        # Check conversation depth
        # Rough heuristic: 4 chars per token
        total_chars = sum(len(str(m.content)) for m in messages)
        if total_chars > 20000:  # Rough approximation of 5000 tokens
            logger.debug("router.classification", tier="COMPLEX", reason="long_history")
            return ModelTier.COMPLEX

        logger.debug("router.classification", tier="STANDARD", reason="default")
        return ModelTier.STANDARD

    def _is_provider_healthy(self, tier: ModelTier) -> bool:
        """Check if a provider is considered healthy based on recent failures."""
        now = time.monotonic()
        # Clean up failures older than 5 minutes (300 seconds)
        recent_failures = [ts for ts in self._provider_failures[tier] if now - ts < 300]
        self._provider_failures[tier] = recent_failures

        # If 3 or more failures in last 5 minutes, consider unhealthy
        if len(recent_failures) >= 3:
            return False

        # Check TTFT (Time To First Token) degradation
        recent_ttft = [t for t in self._provider_ttft[tier][-5:]]
        if len(recent_ttft) >= 3:
            avg_ttft = sum(recent_ttft) / len(recent_ttft)
            if avg_ttft > 20.0:  # 20 seconds is heavily degraded
                logger.warning("router.provider_degraded", tier=tier.value, avg_ttft=avg_ttft)
                return False

        return True

    def _record_ttft(self, tier: ModelTier, ttft_s: float):
        """Record TTFT for a provider."""
        self._provider_ttft[tier].append(ttft_s)
        # Keep only the last 10 entries to avoid memory leak
        self._provider_ttft[tier] = self._provider_ttft[tier][-10:]

    def _record_failure(self, tier: ModelTier):
        """Record a failure for a provider."""
        now = time.monotonic()
        self._provider_failures[tier].append(now)

    def _record_success(self, tier: ModelTier) -> None:
        """Clear transient failures after a successful provider request."""
        self._provider_failures[tier].clear()

    @staticmethod
    def _should_record_failure(error: Exception) -> bool:
        """Only quarantine a provider for availability failures, never bad model output."""
        if isinstance(
            error,
            (AuthenticationError, ProviderUnavailableError, RateLimitError, TimeoutError),
        ):
            return True
        status = getattr(error, "status_code", getattr(error, "status", None))
        if status is None:
            return isinstance(error, (ConnectionError, OSError))
        try:
            status_code = int(status)
        except (TypeError, ValueError):
            return isinstance(error, (ConnectionError, OSError))
        return status_code in {401, 403, 408, 409, 429} or status_code >= 500

    def _get_fallback_tier(self, tier: ModelTier) -> ModelTier | None:
        """Get the next tier to fall back to."""
        if tier == ModelTier.FAST:
            return ModelTier.STANDARD
        if tier == ModelTier.COMPLEX:
            return ModelTier.STANDARD
        if tier == ModelTier.STANDARD:
            return ModelTier.COMPLEX
        return None

    async def route(self, messages: list[Message], **kwargs) -> LLMResponse:
        """Route the query to the optimal model based on intent analysis."""
        target_tier = await self._classify_intent(messages)

        return await self.generate_with_fallback(messages, target_tier=target_tier, **kwargs)

    def _candidates(self, tiers: list[ModelTier], messages: list[Message]):
        visual = has_images(messages)
        if visual and self.vision_fallback:
            yield ModelTier.FAST, self.vision_fallback
        for tier in tiers:
            provider = self.providers.get(tier)
            if provider is not None and (not visual or provider.supports_images is True):
                yield tier, provider

    @staticmethod
    async def _bounded_stream(stream):
        """A stalled cloud connection must not hold the UI indefinitely."""
        try:
            while True:
                try:
                    yield await asyncio.wait_for(anext(stream), timeout=25.0)
                except StopAsyncIteration:
                    return
        finally:
            if hasattr(stream, "aclose"):
                await stream.aclose()

    async def route_stream(
        self,
        messages: list[Message],
        target_tier: ModelTier | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Route and stream the response with fallback on error."""
        if target_tier is None:
            target_tier = await self._classify_intent(messages)

        tiers_to_try = [target_tier]
        fallback = self._get_fallback_tier(target_tier)
        if fallback:
            tiers_to_try.append(fallback)

        if ModelTier.STANDARD not in tiers_to_try:
            tiers_to_try.append(ModelTier.STANDARD)

        if ModelTier.FAST not in tiers_to_try:
            tiers_to_try.append(ModelTier.FAST)

        if ModelTier.COMPLEX not in tiers_to_try:
            tiers_to_try.append(ModelTier.COMPLEX)

        last_error = None

        attempted_provider_ids: set[int] = set()
        for tier, provider in self._candidates(tiers_to_try, messages):
            if kwargs.get("tools") and provider.name != "gemini":
                continue
            if not self._is_provider_healthy(tier):
                logger.warning("router.provider_unhealthy_skip", tier=tier.value)
                continue

            if id(provider) in attempted_provider_ids:
                continue
            attempted_provider_ids.add(id(provider))
            model_name = getattr(provider, "model", "unknown")
            logger.info(
                "router.attempting_stream",
                tier=tier.value,
                provider=provider.name,
                model=model_name,
            )

            chunk_yielded = False
            try:
                stream_iter = provider.stream(messages, **kwargs)

                # Yield metadata for the UI
                yield {"__metadata__": {"provider": provider.name, "model": model_name}}

                ttft_start = time.perf_counter()
                async for chunk in self._bounded_stream(stream_iter):
                    if not chunk_yielded:
                        ttft = time.perf_counter() - ttft_start
                        metrics.record_ttft(ttft)
                        self._record_ttft(tier, ttft)
                        logger.info(
                            "router.stream_success",
                            tier=tier.value,
                            provider=provider.name,
                            model=model_name,
                            ttft=round(ttft, 3),
                        )
                    chunk_yielded = True
                    yield chunk
                if not chunk_yielded:
                    raise ProviderUnavailableError(
                        f"{provider.name} returned an empty streaming response"
                    )
                self._record_success(tier)
                return  # Success
            except Exception as e:
                http_status = getattr(e, "status_code", getattr(e, "status", "unknown"))
                logger.warning(
                    "router.stream_fallback",
                    tier=tier.value,
                    provider=provider.name,
                    model=model_name,
                    reason=str(e),
                    http_status=http_status,
                )
                if self._should_record_failure(e):
                    self._record_failure(tier)
                last_error = e
                if chunk_yielded:
                    # Can't seamlessly fallback if we already output partial text
                    raise LLMError("The reply was interrupted. Please retry.") from e
                    return
                # If nothing yielded yet, loop continues to the next tier fallback
                continue

        raise LLMError(f"All providers failed to stream. Last error: {last_error}")

    async def generate_with_fallback(
        self, messages: list[Message], target_tier: ModelTier | None = None, **kwargs
    ) -> LLMResponse:
        """Try primary tier, fall back to next tier on error."""
        if target_tier is None:
            target_tier = await self._classify_intent(messages)

        tiers_to_try = [target_tier]
        fallback = self._get_fallback_tier(target_tier)
        if fallback:
            tiers_to_try.append(fallback)

        # Add a final fallback just in case
        if ModelTier.STANDARD not in tiers_to_try:
            tiers_to_try.append(ModelTier.STANDARD)

        if ModelTier.FAST not in tiers_to_try:
            tiers_to_try.append(ModelTier.FAST)

        if ModelTier.COMPLEX not in tiers_to_try:
            tiers_to_try.append(ModelTier.COMPLEX)

        last_error = None

        attempted_provider_ids: set[int] = set()
        for tier, provider in self._candidates(tiers_to_try, messages):
            if kwargs.get("tools") and provider.name != "gemini":
                continue
            if not self._is_provider_healthy(tier):
                logger.warning("router.provider_unhealthy_skip", tier=tier.value)
                continue

            if id(provider) in attempted_provider_ids:
                continue
            attempted_provider_ids.add(id(provider))
            model_name = getattr(provider, "model", "unknown")
            logger.info(
                "router.attempting_generate",
                tier=tier.value,
                provider=provider.name,
                model=model_name,
            )

            try:
                start_time = time.perf_counter()
                response = await asyncio.wait_for(
                    provider.generate(messages, **kwargs), timeout=30.0
                )
                if not response.content.strip() and not response.tool_calls:
                    raise ProviderUnavailableError(f"{provider.name} returned an empty response")
                latency_s = time.perf_counter() - start_time
                self._record_success(tier)
                logger.info(
                    "router.generate_success",
                    tier=tier.value,
                    provider=provider.name,
                    model=model_name,
                    latency_s=round(latency_s, 3),
                )
                return response
            except Exception as e:
                http_status = getattr(e, "status_code", getattr(e, "status", "unknown"))
                logger.warning(
                    "router.generate_fallback",
                    tier=tier.value,
                    provider=provider.name,
                    model=model_name,
                    reason=str(e),
                    http_status=http_status,
                )
                if self._should_record_failure(e):
                    self._record_failure(tier)
                last_error = e

        raise LLMError(f"All providers failed. Last error: {last_error}")
