"""Groq provider implementation."""

import asyncio
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
        api_key: str,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ):
        """Initialize the Groq provider."""
        self.client = AsyncGroq(api_key=api_key)
        self._model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def name(self) -> str:
        """Name of the provider."""
        return "groq"

    @property
    def tier(self) -> ModelTier:
        """Model tier of the provider."""
        return ModelTier.FAST

    def _convert_messages(self, messages: list[Message], system_prompt: str | None = None) -> list[dict]:
        """Convert standard messages to Groq/OpenAI format."""
        formatted = []
        
        # Groq Llama models sometimes hallucinate tool calls when there are none, causing API crashes.
        # We enforce a strict plain text rule.
        anti_tool_instruction = "\nCRITICAL SYSTEM INSTRUCTION: DO NOT output any function calls, tool calls, or raw JSON. Respond ONLY in conversational plain text."
        
        if system_prompt:
            formatted.append({"role": "system", "content": system_prompt + anti_tool_instruction})
        else:
            formatted.append({"role": "system", "content": anti_tool_instruction})
            
        for msg in messages:
            role = msg.role
            # Groq will complain if we use 'tool' or 'function' without tools provided
            # or if the roles are non-standard.
            if role not in ("system", "user", "assistant"):
                role = "user"

            if isinstance(msg.content, list):
                # Extract only text parts, ignore images
                text_parts = [item for item in msg.content if isinstance(item, str)]
                formatted.append({"role": role, "content": "\n".join(text_parts)})
            else:
                formatted.append({"role": role, "content": str(msg.content)})
            
        return formatted

    async def _execute_with_retry(self, func, *args, **kwargs):
        """Execute a function with exponential backoff for rate limits."""
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except GroqRateLimitError as e:
                if attempt == max_retries - 1:
                    logger.error("groq.rate_limit_exhausted", error=str(e))
                    raise RateLimitError(f"Rate limit exceeded after {max_retries} retries: {e}")
                
                delay = base_delay * (2 ** attempt)
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
                self.client.chat.completions.create,
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
            "groq.generate.success",
            latency_ms=timer.latency_ms,
            total_tokens=usage.total_tokens
        )

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=self._model_name,
            provider=self.name,
            usage=usage,
            latency_ms=timer.latency_ms,
            finish_reason=response.choices[0].finish_reason or "stop",
            raw=response.model_dump()
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

        try:
            stream = await self._execute_with_retry(
                self.client.chat.completions.create,
                model=self._model_name,
                messages=formatted_msgs,
                temperature=req_temp,
                max_tokens=req_tokens,
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error("groq.stream.error", error=str(e))
            raise

    async def health_check(self) -> bool:
        """Check if Groq API is healthy."""
        try:
            await self.generate(
                messages=[Message.user("ping")],
                max_tokens=5
            )
            return True
        except Exception as e:
            logger.warning("groq.health_check.failed", error=str(e))
            return False
