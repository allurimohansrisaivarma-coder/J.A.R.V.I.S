"""Web search context source."""

import asyncio
import re

import structlog

from jarvis.context.base import ContextSource
from jarvis.context.web_evidence import WEB_UNAVAILABLE, format_evidence, search_evidence

logger = structlog.get_logger(__name__)

_WEATHER_TERMS = ("weather", "climate", "forecast", "temperature")


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
            "appointment",
            "meeting",
            "project",
            "inbox",
            "workday",
        }
        if any(re.search(r"\b" + term + r"s?\b", query_lower) for term in local_terms) or re.search(
            r"\b(my|our|saved|previous) (conversation|chat|memory|name|schedule)\b", query_lower
        ):
            return False

        # Time-sensitive keywords. Generic questions stay local so ordinary or
        # personal prompts are not silently sent to a third-party search engine.
        factual_keywords = [
            "latest",
            "headlines",
            "standings",
            "championship",
            "cricket",
            "ashes",
            "debrief",
            "briefing",
            "news",
            "current",
            "update",
            "today",
            "tomorrow",
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
            "climate",
            "forecast",
            "formula 1",
            "formula one",
            "f1",
            "sprint",
            "grand prix",
            "race time",
            "hackathon",
            "war",
            "conflict",
        ]
        is_factual = any(
            re.search(r"\b" + re.escape(k) + r"\b", query_lower) for k in factual_keywords
        )

        # Explicit commands
        explicit_search = bool(
            re.search(r"\b(search|look up|find|verify|fact check)\b", query_lower)
        )

        return is_factual or explicit_search

    async def gather_context(self, query: str, **kwargs) -> str:
        logger.info("Executing web search", query=query)

        try:
            mcp = kwargs.get("mcp")
            query_lower = query.casefold()
            if mcp is not None and any(term in query_lower for term in _WEATHER_TERMS):
                day = "tomorrow" if "tomorrow" in query_lower else "current"
                city = str(kwargs.get("weather_city") or "").strip()
                try:
                    result = await asyncio.wait_for(
                        mcp.call_tool(
                            "get_weather",
                            {"city_name": city, "day": day},
                        ),
                        timeout=7.0,
                    )
                    if result and not result.startswith(
                        ("Error:", "Error from tool:", "Execution error:", "Failed to get")
                    ):
                        return f"Verified live weather result:\n{result}"
                    logger.warning("Weather MCP unavailable; using web fallback", result=result)
                except TimeoutError:
                    logger.warning("Weather MCP timed out; using web fallback")

            if mcp is not None:
                try:
                    max_results = (
                        10
                        if any(
                            phrase in query_lower
                            for phrase in ("top 10", "top ten", "10 news", "ten news")
                        )
                        else 5
                    )
                    result = await asyncio.wait_for(
                        mcp.call_tool(
                            "web_search",
                            {"query": query, "max_results": max_results},
                        ),
                        timeout=16.0,
                    )
                except TimeoutError:
                    logger.warning("MCP web search timed out after 16 seconds")
                    result = ""
                formatted = format_evidence(result)
                if formatted != WEB_UNAVAILABLE:
                    return formatted
                # A failed MCP retrieval is not evidence. Never pass its error to the model as a result.
                return WEB_UNAVAILABLE

            return format_evidence(
                await asyncio.wait_for(asyncio.to_thread(search_evidence, query), timeout=16.0)
            )
        except Exception as exc:
            logger.warning("Web evidence unavailable", error=str(exc))
            return WEB_UNAVAILABLE

    def _search(self, query: str) -> list[dict]:
        import json

        return json.loads(search_evidence(query)).get("results", [])
