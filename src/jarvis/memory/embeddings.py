"""Embeddings generation using Gemini."""

import structlog
from google import genai
from google.genai import types
from jarvis.config.settings import get_settings

logger = structlog.get_logger(__name__)

class Embedder:
    """Generates vector embeddings for text."""
    
    def __init__(self, api_keys: str | list[str] | None = None, model: str = "gemini-embedding-2"):
        """Initialize the embedder.
        
        Args:
            api_keys: Gemini API keys. If not provided, fetched from settings.
            model: The Gemini embedding model to use.
        """
        self.model = model
        if not api_keys:
            api_keys = get_settings().gemini_api_keys
            
        if not api_keys or (isinstance(api_keys, str) and api_keys in ("", "fallback", "env_or_placeholder")):
            raise ValueError("GEMINI_API_KEYS must be set for embeddings.")
        
        self.api_keys = api_keys if isinstance(api_keys, list) else [api_keys]
        self._current_key_idx = 0
        self.client = genai.Client(api_key=self.api_keys[self._current_key_idx])
        
    def _execute_with_retry(self, task_type: str, text: str) -> list[float]:
        import time
        from google.genai import errors
        
        max_retries = max(3, len(self.api_keys) * 2)
        base_delay = 1.0
        keys_attempted_this_request = 1

        for attempt in range(max_retries):
            try:
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=text,
                    config=types.EmbedContentConfig(task_type=task_type)
                )
                if response.embeddings and len(response.embeddings) > 0:
                    return response.embeddings[0].values
                return []
            except errors.ClientError as e:
                if e.code == 429:
                    if keys_attempted_this_request < len(self.api_keys):
                        self._current_key_idx = (self._current_key_idx + 1) % len(self.api_keys)
                        logger.warning("embeddings.rate_limit_rotating_key", key_idx=self._current_key_idx)
                        self.client = genai.Client(api_key=self.api_keys[self._current_key_idx])
                        keys_attempted_this_request += 1
                        continue

                    # All keys exhausted, zero-latency failover
                    logger.error("embeddings.rate_limit_exhausted", error=str(e))
                    return []
                else:
                    logger.error("Failed to generate embedding", error=str(e))
                    return []
            except Exception as e:
                logger.error("Failed to generate embedding", error=str(e))
                return []
        return []

    def embed_text(self, text: str) -> list[float]:
        """Generate a vector embedding for a single string of text to be stored."""
        return self._execute_with_retry("RETRIEVAL_DOCUMENT", text)
            
    def embed_query(self, text: str) -> list[float]:
        """Generate a vector embedding for a search query."""
        return self._execute_with_retry("RETRIEVAL_QUERY", text)
