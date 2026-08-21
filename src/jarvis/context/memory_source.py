"""Memory context source."""

import asyncio
from typing import TYPE_CHECKING

import structlog

from jarvis.context.base import ContextSource

if TYPE_CHECKING:
    from jarvis.memory.manager import MemoryManager

logger = structlog.get_logger(__name__)


class MemoryContextSource(ContextSource):
    """Injects relevant facts from the memory system."""

    def __init__(self, memory_manager: "MemoryManager | None" = None):
        self.memory = memory_manager

    @property
    def name(self) -> str:
        return "Long-term Memory"

    async def can_handle(self, query: str) -> bool:
        """Memory is almost always relevant, but we only query if the memory system is online."""
        return self.memory is not None

    async def gather_context(self, query: str, **kwargs) -> str:
        if not self.memory:
            return ""

        search_query = query
        router = kwargs.get("router")

        if router:
            from jarvis.llm.base import Message, ModelTier

            try:
                expansion = await router.generate_with_fallback(
                    [
                        Message.user(
                            "You are a search query optimizer. The user is asking a question or making a statement. "
                            "Rewrite it into 1-2 concise declarative statements representing the facts the user is asking about or storing, to optimize for vector database retrieval. "
                            "Output ONLY the optimized search string without quotes or preamble.\n\n"
                            f"User: {query}"
                        )
                    ],
                    target_tier=ModelTier.FAST,
                )
                expanded = expansion.content.strip().strip("\"'")
                if expanded:
                    search_query = expanded
                    logger.debug(
                        "Query expanded for memory search", original=query, expanded=search_query
                    )
            except Exception as e:
                logger.warning("Query expansion failed", error=str(e))

        try:
            results = await asyncio.to_thread(self.memory.semantic_search, search_query, limit=50)
            # Filter results by a reasonable distance threshold (L2 distance up to ~1.414 is orthogonal, but conversational queries can be distant)
            relevant = [r for r in results if r["distance"] < 1.5]

            if not relevant:
                return ""

            unique_facts = []
            seen = set()
            for r in relevant:
                text = r["content"].strip()
                timestamp = r.get("timestamp", "Unknown")
                if isinstance(timestamp, str) and "T" in timestamp:
                    timestamp = timestamp.split("T")[0]  # just use the date or format it

                if text.lower() not in seen:
                    seen.add(text.lower())
                    unique_facts.append(f"- [{timestamp}] {text}")
                    if len(unique_facts) >= 20:
                        break

            formatted = "\n".join(unique_facts)
            return f"Relevant memories about the user or past conversations:\n{formatted}"
        except Exception as e:
            logger.error("Failed to gather memory context", error=str(e))
            return ""
