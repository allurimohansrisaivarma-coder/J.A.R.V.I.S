"""Web search context source."""

import asyncio

import structlog
from ddgs import DDGS

from jarvis.context.base import ContextSource

logger = structlog.get_logger(__name__)


class WebContextSource(ContextSource):
    """Searches the web for real-time information."""

    @property
    def name(self) -> str:
        return "Web Search"

    async def can_handle(self, query: str) -> bool:
        """Determines if the query likely needs a web search."""
        query_lower = query.lower()

        # Never send private local-file or calendar requests to web search.
        local_terms = {
            "file",
            "folder",
            "directory",
            "desktop",
            "document",
            "download",
            "calendar",
            "schedule",
            "appointment",
        }
        if any(term in query_lower for term in local_terms):
            return False

        # Time-sensitive keywords. Generic questions stay local so ordinary or
        # personal prompts are not silently sent to a third-party search engine.
        factual_keywords = [
            "latest",
            "news",
            "current",
            "update",
            "today",
            "now",
            "2024",
            "2025",
            "2026",
            "2027",
            "2028",
            "2029",
            "2030",
            "season",
            "stats",
            "score",
            "price",
            "weather",
            "hackathon",
            "war",
            "conflict",
        ]
        is_factual = any(k in query_lower for k in factual_keywords)

        # Explicit commands
        explicit_search = (
            "search" in query_lower or "look up" in query_lower or "find" in query_lower
        )

        return is_factual or explicit_search

    async def gather_context(self, query: str, **kwargs) -> str:
        logger.info("Executing web search", query=query)

        try:
            mcp = kwargs.get("mcp")
            if mcp is not None:
                try:
                    result = await asyncio.wait_for(
                        mcp.call_tool(
                            "web_search",
                            {"query": query, "max_results": 5},
                        ),
                        timeout=4.0,
                    )
                except TimeoutError:
                    logger.warning("MCP web search timed out after 4 seconds")
                    return ""
                if result and not result.startswith(
                    ("Error:", "Error from tool:", "Execution error:")
                ):
                    return f"Current Web Search Results for '{query}':\n{result}"
                logger.warning("MCP web search unavailable; using direct fallback", result=result)

            # DDGS is synchronous, so we run it in a thread pool
            results = await asyncio.wait_for(asyncio.to_thread(self._search, query), timeout=4.0)

            if not results:
                return ""

            formatted = "\n".join(
                [
                    f"- {r.get('title', 'Unknown')}: {r.get('body', 'No description')} (Source: {r.get('href', 'N/A')})"
                    for r in results[:5]
                ]
            )
            return f"Current Web Search Results for '{query}':\n{formatted}"
        except TimeoutError:
            logger.warning("Direct web search timed out after 4 seconds")
            return ""
        except Exception as e:
            logger.error("Web search failed", error=str(e))
            return ""

    def _search(self, query: str) -> list[dict]:
        try:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=5))
        except Exception as e:
            logger.warning("DuckDuckGo search exception", error=str(e))
            return []
