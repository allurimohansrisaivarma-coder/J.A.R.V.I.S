"""Model routing layer."""

import asyncio
import time
from typing import AsyncIterator, Dict

import structlog

from jarvis.config.settings import RouterSettings
from jarvis.llm.base import (
    LLMError,
    LLMProvider,
    LLMResponse,
    Message,
    ModelTier,
)

logger = structlog.get_logger(__name__)

class ModelRouter:
    """Routes LLM requests to appropriate models based on intent and load."""

    def __init__(self, providers: Dict[ModelTier, LLMProvider], settings: RouterSettings):
        """Initialize the router."""
        self.providers = providers
        self.settings = settings
        # Track provider failures (dict of tier to list of failure timestamps)
        self._provider_failures: Dict[ModelTier, list[float]] = {tier: [] for tier in ModelTier}
        
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
            if keyword.lower() in last_msg.split():
                if word_count < 20 and not any(tech_term in last_msg for tech_term in ["code", "function", "class", "def ", "class "]):
                    logger.debug("router.classification", tier="FAST", reason="fast_keyword")
                    return ModelTier.FAST

        # Simple conversational signals
        fast_signals = {"yes", "no", "ok", "thanks", "go ahead", "sure", "yep", "nope"}
        if last_msg.strip() in fast_signals:
            logger.debug("router.classification", tier="FAST", reason="simple_response")
            return ModelTier.FAST

        # Check for complex intent
        complex_keywords = ["analyze", "explain in detail", "compare", "architecture"]
        if any(keyword in last_msg for keyword in complex_keywords):
            logger.debug("router.classification", tier="COMPLEX", reason="complex_keyword")
            return ModelTier.COMPLEX

        # Check conversation depth
        # Rough heuristic: 4 chars per token
        total_chars = sum(len(str(m.content)) for m in messages)
        if total_chars > 20000: # Rough approximation of 5000 tokens
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
        return len(recent_failures) < 3

    def _record_failure(self, tier: ModelTier):
        """Record a failure for a provider."""
        now = time.monotonic()
        self._provider_failures[tier].append(now)

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
        
        if target_tier not in self.providers:
            # Fallback if preferred tier isn't configured
            logger.warning("router.missing_provider", tier=target_tier.value)
            target_tier = ModelTier.STANDARD
            
        provider = self.providers[target_tier]
        return await provider.generate(messages, **kwargs)

    async def route_stream(self, messages: list[Message], **kwargs) -> AsyncIterator[str]:
        """Route and stream the response with fallback on error."""
        target_tier = await self._classify_intent(messages)
        
        tiers_to_try = [target_tier]
        fallback = self._get_fallback_tier(target_tier)
        if fallback:
            tiers_to_try.append(fallback)
            
        if ModelTier.STANDARD not in tiers_to_try:
            tiers_to_try.append(ModelTier.STANDARD)
            
        if ModelTier.FAST not in tiers_to_try:
            tiers_to_try.append(ModelTier.FAST)

        last_error = None
        
        for tier in tiers_to_try:
            if tier not in self.providers:
                continue
                
            if not self._is_provider_healthy(tier):
                logger.warning("router.provider_unhealthy_skip", tier=tier.value)
                continue
                
            provider = self.providers[tier]
            logger.info("router.attempting_stream", tier=tier.value, provider=provider.name)
            
            try:
                stream_iter = provider.stream(messages, **kwargs)
                chunk_yielded = False
                async for chunk in stream_iter:
                    chunk_yielded = True
                    yield chunk
                return  # Success
            except Exception as e:
                logger.error("router.stream_failed", tier=tier.value, error=str(e))
                self._record_failure(tier)
                last_error = e
                if chunk_yielded:
                    # Can't seamlessly fallback if we already output partial text
                    yield f"\n\n[Error: Connection interrupted. {e}]"
                    return
                # If nothing yielded yet, loop continues to the next tier fallback
                continue
                
        raise LLMError(f"All providers failed to stream. Last error: {last_error}")

    async def generate_with_fallback(self, messages: list[Message], **kwargs) -> LLMResponse:
        """Try primary tier, fall back to next tier on error."""
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

        last_error = None
        
        for tier in tiers_to_try:
            if tier not in self.providers:
                continue
                
            if not self._is_provider_healthy(tier):
                logger.warning("router.provider_unhealthy_skip", tier=tier.value)
                continue
                
            provider = self.providers[tier]
            logger.info("router.attempting_generation", tier=tier.value, provider=provider.name)
            
            try:
                response = await provider.generate(messages, **kwargs)
                return response
            except Exception as e:
                logger.error("router.generation_failed", tier=tier.value, error=str(e))
                self._record_failure(tier)
                last_error = e
                
        raise LLMError(f"All providers failed. Last error: {last_error}")
