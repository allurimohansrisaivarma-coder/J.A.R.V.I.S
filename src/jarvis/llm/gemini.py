"""Gemini provider implementation."""

from collections.abc import AsyncIterator
from typing import Any

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
        api_keys: str | list[str] | None = None,
        model: str = "gemini-1.5-flash",
        temperature: float = 0.7,
        max_output_tokens: int = 2048,
        thinking_level: str | None = None,
        *,
        api_key: str | None = None,
    ):
        """Initialize the Gemini provider."""
        if api_keys is None and api_key:
            api_keys = api_key
        if not api_keys:
            raise ValueError("At least one API key must be provided for GeminiProvider")
        self.api_keys = api_keys if isinstance(api_keys, list) else [api_keys]
        self._current_key_idx = 0
        self._model_name = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.thinking_level = thinking_level
        self.client = self._create_client()

    def _create_client(self):
        return genai.Client(
            api_key=self.api_keys[self._current_key_idx],
            http_options=types.HttpOptions(
                timeout=20000, retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )

    @property
    def supports_images(self) -> bool:
        return True

    @property
    def name(self) -> str:
        """Name of the provider."""
        return "gemini"

    @property
    def model(self) -> str:
        """Configured Gemini model identifier."""
        return self._model_name

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
                        elif isinstance(item, types.Part):
                            parts.append(item)
                        else:
                            # Assume it's a PIL Image
                            import io

                            img_byte_arr = io.BytesIO()
                            # Resize if huge to save bandwidth (if not already resized)
                            img = item.convert("RGB")
                            img.thumbnail((1920, 1200))
                            img.save(img_byte_arr, format="JPEG", quality=85)
                            img_bytes = img_byte_arr.getvalue()
                            parts.append(
                                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
                            )
                    contents.append(types.Content(role="user", parts=parts))
                else:
                    contents.append(
                        types.Content(
                            role="user", parts=[types.Part.from_text(text=str(msg.content))]
                        )
                    )
            elif msg.role == "assistant":
                if isinstance(msg.content, list):
                    parts = []
                    for item in msg.content:
                        if isinstance(item, str):
                            parts.append(types.Part.from_text(text=item))
                        elif isinstance(item, types.Part):
                            parts.append(item)
                        else:
                            parts.append(types.Part.from_text(text=str(item)))
                    contents.append(types.Content(role="model", parts=parts))
                else:
                    contents.append(
                        types.Content(
                            role="model", parts=[types.Part.from_text(text=str(msg.content))]
                        )
                    )
            elif msg.role == "tool" and isinstance(msg.content, list):
                parts = [item for item in msg.content if isinstance(item, types.Part)]
                contents.append(types.Content(role="user", parts=parts))

        return contents, system_prompt

    async def _execute_with_retry(self, method_name: str, *args, **kwargs):
        """Execute a method on self.client.aio.models with key rotation and backoff."""
        max_retries = max(3, len(self.api_keys) * 2)
        keys_attempted_this_request = 1

        for attempt in range(max_retries):
            try:
                func = getattr(self.client.aio.models, method_name)
                return await func(*args, **kwargs)
            except errors.APIError as e:
                # 429 Too Many Requests, 503 Service Unavailable, 500 Internal Error
                if e.code in (429, 503, 500):
                    # Key rotation logic
                    if e.code == 429 and keys_attempted_this_request < len(self.api_keys):
                        self._current_key_idx = (self._current_key_idx + 1) % len(self.api_keys)
                        logger.warning(
                            "gemini.rate_limit_rotating_key", key_idx=self._current_key_idx
                        )
                        self.client = self._create_client()
                        keys_attempted_this_request += 1
                        continue  # Immediate retry with new key

                    # All keys exhausted, zero-latency failover
                    logger.error("gemini.rate_limit_exhausted", error=str(e))
                    raise RateLimitError(
                        f"Rate limit exceeded across all {len(self.api_keys)} keys: {e}"
                    )
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
        tools: Any | None = None,
    ) -> LLMResponse:
        """Generate response with retry logic and telemetry."""
        contents, msg_sys_prompt = self._convert_messages(messages)

        final_system_prompt = system_prompt or msg_sys_prompt

        configured_tools = tools if isinstance(tools, list) else ([tools] if tools else None)
        uses_current_thinking_api = self._model_name.startswith(("gemini-3.6", "gemini-3.7"))
        config = types.GenerateContentConfig(
            temperature=(
                None
                if uses_current_thinking_api
                else (temperature if temperature is not None else self.temperature)
            ),
            max_output_tokens=max_tokens if max_tokens is not None else self.max_output_tokens,
            system_instruction=final_system_prompt,
            tools=configured_tools,
            thinking_config=(
                types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel(self.thinking_level.upper())
                )
                if uses_current_thinking_api and self.thinking_level
                else None
            ),
            # JARVIS owns the multi-turn tool loop. The SDK should return
            # function calls without executing them automatically.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        logger.debug(
            "gemini.generate.start",
            model=self._model_name,
            message_count=len(messages),
            key_idx=self._current_key_idx + 1,
        )

        with self._build_timing_context() as timer:
            response = await self._execute_with_retry(
                "generate_content",
                model=self._model_name,
                contents=contents,
                config=config,
            )

        usage = TokenUsage()
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage.prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            usage.completion_tokens = (
                getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            )

        logger.info(
            "gemini.generate.success",
            latency_ms=timer.latency_ms,
            total_tokens=usage.total_tokens,
            key_idx=self._current_key_idx + 1,
        )

        tool_calls = []
        if (
            response.candidates
            and response.candidates[0].content
            and response.candidates[0].content.parts
        ):
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    tool_calls.append(part.function_call)

        text_parts = []
        if response.candidates and response.candidates[0].content:
            text_parts = [
                part.text
                for part in response.candidates[0].content.parts or []
                if part.text and not part.thought
            ]

        return LLMResponse(
            content="".join(text_parts),
            model=self._model_name,
            provider=self.name,
            usage=usage,
            latency_ms=timer.latency_ms,
            finish_reason=str(response.candidates[0].finish_reason.name)
            if response.candidates and response.candidates[0].finish_reason
            else "stop",
            raw=response.model_dump(),
            tool_calls=tool_calls,
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        tools: Any | None = None,
    ) -> AsyncIterator[Any]:
        """Stream response chunks from the model."""
        contents, msg_sys_prompt = self._convert_messages(messages)

        final_system_prompt = system_prompt or msg_sys_prompt

        configured_tools = tools if isinstance(tools, list) else ([tools] if tools else None)
        uses_current_thinking_api = self._model_name.startswith(("gemini-3.6", "gemini-3.7"))
        config = types.GenerateContentConfig(
            temperature=(
                None
                if uses_current_thinking_api
                else (temperature if temperature is not None else self.temperature)
            ),
            max_output_tokens=max_tokens if max_tokens is not None else self.max_output_tokens,
            system_instruction=final_system_prompt,
            tools=configured_tools,
            thinking_config=(
                types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel(self.thinking_level.upper())
                )
                if uses_current_thinking_api and self.thinking_level
                else None
            ),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        logger.debug(
            "gemini.stream.start", model=self._model_name, key_idx=self._current_key_idx + 1
        )

        max_retries = max(3, len(self.api_keys) * 2)
        keys_attempted_this_request = 1

        emitted = False
        for attempt in range(max_retries):
            try:
                # Need to grab the current client in case it rotated
                stream = await self.client.aio.models.generate_content_stream(
                    model=self._model_name,
                    contents=contents,
                    config=config,
                )
                async for chunk in stream:
                    if chunk.parts:
                        for p in chunk.parts:
                            if p.text and not p.thought:
                                emitted = True
                                yield p.text
                            elif p.function_call:
                                emitted = True
                                yield p
                return  # Success, exit the retry loop
            except Exception as e:
                if isinstance(e, errors.APIError) and e.code in (429, 503, 500):
                    if (
                        not emitted
                        and e.code == 429
                        and keys_attempted_this_request < len(self.api_keys)
                    ):
                        self._current_key_idx = (self._current_key_idx + 1) % len(self.api_keys)
                        logger.warning(
                            "gemini.rate_limit_rotating_key_in_stream",
                            key_idx=self._current_key_idx,
                        )
                        self.client = self._create_client()
                        keys_attempted_this_request += 1
                        continue
                    logger.error("gemini.rate_limit_exhausted_stream", error=str(e))
                    raise RateLimitError(
                        f"Rate limit exceeded across all {len(self.api_keys)} keys: {e}"
                    ) from e
                else:
                    logger.error("gemini.stream.error", error=str(e))
                    raise

    async def health_check(self) -> bool:
        """Check if the Gemini API is reachable."""
        try:
            await self._execute_with_retry("get", model=self._model_name)
            return True
        except Exception as e:
            logger.warning("gemini.health_check.failed", error=str(e))
            return False
