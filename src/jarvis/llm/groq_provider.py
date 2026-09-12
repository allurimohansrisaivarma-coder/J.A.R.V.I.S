"""Groq provider implementation."""

import asyncio
import base64
import io
from collections.abc import AsyncIterator
from typing import Any

import structlog
from groq import APIStatusError, AsyncGroq
from groq import AuthenticationError as GroqAuthError
from groq import RateLimitError as GroqRateLimitError

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
)

logger = structlog.get_logger(__name__)


class GroqProvider(LLMProvider):
    """Groq LLM Provider."""

    def __init__(
        self,
        api_keys: list[str] | None = None,
        model: str = "openai/gpt-oss-20b",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        *,
        api_key: str | None = None,
        supports_images: bool = False,
    ):
        """Initialize the Groq provider."""
        if api_keys is None and api_key:
            api_keys = [api_key]
        if not api_keys:
            raise ValueError("At least one API key must be provided for GroqProvider")
        self.api_keys = api_keys
        self._current_key_idx = 0
        self.client = AsyncGroq(
            api_key=self.api_keys[self._current_key_idx], timeout=20.0, max_retries=0
        )
        self._supports_images = supports_images
        self._model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def supports_images(self) -> bool:
        return self._supports_images

    @property
    def name(self) -> str:
        """Name of the provider."""
        return "groq"

    @property
    def model(self) -> str:
        """Configured Groq model identifier."""
        return self._model_name

    @property
    def tier(self) -> ModelTier:
        """Model tier of the provider."""
        return ModelTier.FAST

    def _convert_messages(
        self, messages: list[Message], system_prompt: str | None = None
    ) -> list[dict]:
        """Convert standard messages to Groq/OpenAI format."""
        system_messages = []
        dialogue_messages = []

        if system_prompt:
            system_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            role = msg.role
            # Groq will complain if we use 'tool' or 'function' without tools provided
            # or if the roles are non-standard.
            if role not in ("system", "user", "assistant"):
                role = "user"

            if isinstance(msg.content, list):
                from PIL import Image

                text_parts = [item for item in msg.content if isinstance(item, str)]
                converted: dict[str, Any] = {"role": role, "content": "\n".join(text_parts)}
                images = [item for item in msg.content if isinstance(item, Image.Image)]
                if images:
                    if not self.supports_images:
                        raise ProviderUnavailableError("This model cannot read screenshots.")
                    parts: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(text_parts)}]
                    for original in images:
                        img = original.convert("RGB")
                        img.thumbnail((1920, 1200))
                        buffer = io.BytesIO()
                        img.save(buffer, format="JPEG", quality=85)
                        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                            }
                        )
                    converted["content"] = parts
            else:
                converted = {"role": role, "content": str(msg.content)}
            if role == "system":
                # The shared prompt also describes Gemini-only/native tools.
                # Leaving those declarations in a Groq request can make
                # GPT-OSS emit a hidden function call even with tool_choice=none.
                lines = str(converted["content"]).splitlines()
                converted["content"] = "\n".join(
                    line
                    for line in lines
                    if not any(
                        marker in line.casefold()
                        for marker in (
                            "tool",
                            "`launch_application`",
                            "`browser_navigate`",
                            "`browser_search`",
                            "`browser_read_page`",
                            "`browser_click`",
                            "`get_weather`",
                            "`delete_memory`",
                            "use an available tool",
                            "if you have a tool",
                        )
                    )
                )
                system_messages.append(converted)
            else:
                dialogue_messages.append(converted)

        # Keep this as the final system instruction. GPT-OSS may otherwise emit
        # an internal tool call merely because the shared JARVIS prompt describes
        # tools, even though this request has no callable declarations.
        system_messages.append(
            {
                "role": "system",
                "content": (
                    "Provider mode: respond with ordinary text only. No callable tools are attached "
                    "to this request. Never emit a function call or tool call. If the system supplied "
                    "a verified result, summarize that result without attempting the action again."
                ),
            }
        )

        return system_messages + dialogue_messages

    async def _execute_with_retry(self, method_name: str, *args, **kwargs):
        """Execute a function with exponential backoff and key rotation for rate limits."""
        if self._model_name.startswith("qwen/"):
            kwargs.setdefault("reasoning_effort", "none")
        # Max retries should allow cycling through all keys at least twice
        max_retries = max(3, len(self.api_keys) * 2)
        base_delay = 1.0
        keys_attempted_this_request = 1

        for attempt in range(max_retries):
            try:
                func = getattr(self.client.chat.completions, method_name)
                return await func(*args, **kwargs)
            except GroqRateLimitError as e:
                if "request too large" in str(e).lower():
                    raise RateLimitError("The request exceeds this model's token allowance.") from e
                # Key rotation logic
                if keys_attempted_this_request < len(self.api_keys):
                    self._current_key_idx = (self._current_key_idx + 1) % len(self.api_keys)
                    logger.warning("groq.rate_limit_rotating_key", key_idx=self._current_key_idx)
                    self.client = AsyncGroq(
                        api_key=self.api_keys[self._current_key_idx], timeout=20.0, max_retries=0
                    )
                    keys_attempted_this_request += 1
                    continue  # Immediate retry with new key

                # All keys exhausted, apply backoff
                if attempt == max_retries - 1:
                    logger.error("groq.rate_limit_exhausted", error=str(e))
                    raise RateLimitError(
                        f"Rate limit exceeded across all {len(self.api_keys)} keys after {max_retries} retries: {e}"
                    )

                # Reset counter for the next round of retries
                keys_attempted_this_request = 1
                delay = base_delay * (2**attempt)
                logger.warning("groq.rate_limit_retry", attempt=attempt, delay=delay)
                await asyncio.sleep(delay)
            except GroqAuthError as e:
                logger.error("groq.auth_error", error=str(e))
                raise AuthenticationError(f"Authentication failed: {e}")
            except APIStatusError as e:
                logger.error("groq.api_error", error=str(e))
                raise ProviderUnavailableError(f"Groq API error: {e}")
            except Exception as e:
                logger.error("groq.unknown_error", error=str(e))
                raise LLMError(f"Unexpected error calling Groq: {e}")

    async def generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        tools: Any | None = None,
    ) -> LLMResponse:
        """Generate response from Groq."""
        formatted_msgs = self._convert_messages(messages, system_prompt)

        req_temp = temperature if temperature is not None else self.temperature
        req_tokens = max_tokens if max_tokens is not None else self.max_tokens

        logger.debug("groq.generate.start", model=self._model_name, messages=len(formatted_msgs))

        with self._build_timing_context() as timer:
            response = await self._execute_with_retry(
                "create",
                model=self._model_name,
                messages=formatted_msgs,
                temperature=req_temp,
                max_tokens=req_tokens,
                stream=False,
            )

        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
        )

        logger.info(
            "groq.generate.success", latency_ms=timer.latency_ms, total_tokens=usage.total_tokens
        )

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=self._model_name,
            provider=self.name,
            usage=usage,
            latency_ms=timer.latency_ms,
            finish_reason=response.choices[0].finish_reason or "stop",
            raw=response.model_dump(),
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        tools: Any | None = None,
    ) -> AsyncIterator[str]:
        """Stream a response from Groq."""
        formatted_msgs = self._convert_messages(messages, system_prompt)

        req_temp = temperature if temperature is not None else self.temperature
        req_tokens = max_tokens if max_tokens is not None else self.max_tokens

        logger.debug("groq.stream.start", model=self._model_name)

        for attempt in range(2):
            try:
                request_messages = list(formatted_msgs)
                if attempt:
                    request_messages.insert(
                        0,
                        {
                            "role": "system",
                            "content": (
                                "Return a direct plain-text answer. Function and tool syntax is invalid "
                                "for this request, including analysis-channel tool calls."
                            ),
                        },
                    )
                stream = await self._execute_with_retry(
                    "create",
                    model=self._model_name,
                    messages=request_messages,
                    temperature=0 if attempt else req_temp,
                    max_tokens=req_tokens,
                    stream=True,
                )

                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return
            except Exception as e:
                tool_call_error = "tool choice is none" in str(e).casefold()
                if tool_call_error and attempt == 0:
                    logger.warning("groq.stream.retrying_forbidden_tool_call")
                    continue
                logger.error("groq.stream.error", error=str(e))
                raise

    async def health_check(self) -> bool:
        """Check if Groq API is healthy."""
        try:
            await self.generate(messages=[Message.user("ping")], max_tokens=5)
            return True
        except Exception as e:
            logger.warning("groq.health_check.failed", error=str(e))
            return False
