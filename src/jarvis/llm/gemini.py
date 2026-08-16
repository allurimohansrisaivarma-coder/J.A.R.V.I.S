"""Gemini provider implementation."""

import asyncio
from typing import AsyncIterator

import structlog
from google import genai
from google.genai import errors, types

from jarvis.llm.base import (
    LLMError,
    LLMProvider,
    LLMResponse,
    Message,
    ModelTier,
    ProviderUnavailableError,
    RateLimitError,
    TokenUsage,
)

logger = structlog.get_logger(__name__)

class GeminiProvider(LLMProvider):
    """Gemini LLM Provider using google-genai."""

    def __init__(
        self,
        api_keys: str | list[str],
        model: str = "gemini-1.5-flash",
        temperature: float = 0.7,
        max_output_tokens: int = 2048,
    ):
        """Initialize the Gemini provider."""
        self.api_keys = api_keys if isinstance(api_keys, list) else [api_keys]
        self._current_key_idx = 0
        self._model_name = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.client = genai.Client(api_key=self.api_keys[self._current_key_idx])

    @property
    def name(self) -> str:
        """Name of the provider."""
        return "gemini"

    @property
    def tier(self) -> ModelTier:
        """Determine tier based on model name."""
        if "pro" in self._model_name.lower():
            return ModelTier.COMPLEX
        return ModelTier.STANDARD

    def _convert_messages(self, messages: list[Message]) -> tuple[list[types.Content], str | None]:
        """Convert standard messages to Gemini contents."""
        contents = []
        system_prompt = None

        for msg in messages:
            if msg.role == "system":
                # Collect system prompts if multiple, joining with newlines
                if system_prompt:
                    system_prompt += f"\n{msg.content}"
                else:
                    system_prompt = str(msg.content)
            elif msg.role == "user":
                if isinstance(msg.content, list):
                    parts = []
                    for item in msg.content:
                        if isinstance(item, str):
                            parts.append(types.Part.from_text(text=item))
                        else:
                            # Assume it's a PIL Image
                            import io
                            img_byte_arr = io.BytesIO()
                            # Resize if huge to save bandwidth (if not already resized)
                            if item.width > 1920 or item.height > 1080:
                                item.thumbnail((1920, 1080))
                            item.save(img_byte_arr, format='JPEG', quality=80)
                            img_bytes = img_byte_arr.getvalue()
                            parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
                    contents.append(types.Content(role="user", parts=parts))
                else:
                    contents.append(
                        types.Content(role="user", parts=[types.Part.from_text(text=str(msg.content))])
                    )
            elif msg.role == "assistant":
                if isinstance(msg.content, list):
                    parts = [types.Part.from_text(text=str(item)) for item in msg.content]
                    contents.append(types.Content(role="model", parts=parts))
                else:
                    contents.append(
                        types.Content(role="model", parts=[types.Part.from_text(text=str(msg.content))])
                    )

        return contents, system_prompt

    async def _execute_with_retry(self, method_name: str, *args, **kwargs):
        """Execute a method on self.client.aio.models with key rotation and backoff."""
        max_retries = max(3, len(self.api_keys) * 2)
        base_delay = 1.0
        keys_attempted_this_request = 1

        for attempt in range(max_retries):
            try:
                func = getattr(self.client.aio.models, method_name)
                return await func(*args, **kwargs)
            except errors.ClientError as e:
                # 429 Too Many Requests
                if e.code == 429:
                    # Key rotation logic
                    if keys_attempted_this_request < len(self.api_keys):
                        self._current_key_idx = (self._current_key_idx + 1) % len(self.api_keys)
                        logger.warning("gemini.rate_limit_rotating_key", key_idx=self._current_key_idx)
                        self.client = genai.Client(api_key=self.api_keys[self._current_key_idx])
                        keys_attempted_this_request += 1
                        continue  # Immediate retry with new key

                    # All keys exhausted, zero-latency failover
                    logger.error("gemini.rate_limit_exhausted", error=str(e))
                    raise RateLimitError(f"Rate limit exceeded across all {len(self.api_keys)} keys: {e}")
                else:
                    logger.error("gemini.client_error", error=str(e), code=e.code)
                    raise ProviderUnavailableError(f"Gemini API error: {e}")
            except Exception as e:
                logger.error("gemini.unknown_error", error=str(e))
                raise LLMError(f"Unexpected error calling Gemini: {e}")

    async def generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate response with retry logic and telemetry."""
        contents, msg_sys_prompt = self._convert_messages(messages)
        
        final_system_prompt = system_prompt or msg_sys_prompt

        config = types.GenerateContentConfig(
            temperature=temperature if temperature is not None else self.temperature,
            max_output_tokens=max_tokens if max_tokens is not None else self.max_output_tokens,
            system_instruction=final_system_prompt,
        )

        logger.debug("gemini.generate.start", model=self._model_name, message_count=len(messages), key_idx=self._current_key_idx + 1)
        
        with self._build_timing_context() as timer:
            response = await self._execute_with_retry(
                "generate_content",
                model=self._model_name,
                contents=contents,
                config=config,
            )

        usage = TokenUsage()
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage.prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
            usage.completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)

        logger.info(
            "gemini.generate.success", 
            latency_ms=timer.latency_ms,
            total_tokens=usage.total_tokens,
            key_idx=self._current_key_idx + 1
        )

        return LLMResponse(
            content=response.text or "",
            model=self._model_name,
            provider=self.name,
            usage=usage,
            latency_ms=timer.latency_ms,
            # Using basic default string for finish_reason if unavailable
            raw={}
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream response chunks from the model."""
        contents, msg_sys_prompt = self._convert_messages(messages)
        
        final_system_prompt = system_prompt or msg_sys_prompt

        config = types.GenerateContentConfig(
            temperature=temperature if temperature is not None else self.temperature,
            max_output_tokens=max_tokens if max_tokens is not None else self.max_output_tokens,
            system_instruction=final_system_prompt,
        )

        logger.debug("gemini.stream.start", model=self._model_name, key_idx=self._current_key_idx + 1)

        try:
            stream = await self._execute_with_retry(
                "generate_content_stream",
                model=self._model_name,
                contents=contents,
                config=config,
            )
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error("gemini.stream.error", error=str(e))
            raise

    async def health_check(self) -> bool:
        """Check if the Gemini API is reachable."""
        try:
            await self.generate(
                messages=[Message.user("ping")],
                max_tokens=5
            )
            return True
        except Exception as e:
            logger.warning("gemini.health_check.failed", error=str(e))
            return False
