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
        local_terms = {"file", "folder", "directory", "desktop", "document", "download", "calendar", "schedule", "appointment"}
        if any(term in query_lower for term in local_terms):
            return False
        
        # 1. Direct questions
        question_words = ["what", "who", "where", "when", "why", "how"]
        is_question = any(query_lower.startswith(w) or f" {w} " in query_lower for w in question_words) or "?" in query_lower
        
        # 2. Time-sensitive / factual keywords (including future years past LLM cutoff)
        factual_keywords = [
            "latest", "news", "current", "update", "today", "now", 
            "2024", "2025", "2026", "2027", "2028", "2029", "2030",
            "season", "stats", "score", "price", "weather", "hackathon", "war", "conflict"
        ]
        is_factual = any(k in query_lower for k in factual_keywords)
        
        # 3. Explicit commands
        explicit_search = "search" in query_lower or "look up" in query_lower or "find" in query_lower
        
        return is_question or is_factual or explicit_search
        
    async def gather_context(self, query: str, **kwargs) -> str:
        logger.info("Executing web search", query=query)
        
        try:
            # DDGS is synchronous, so we run it in a thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self._search, query)
            
            if not results:
                return ""
                
            formatted = "\n".join([f"- {r.get('title', 'Unknown')}: {r.get('body', 'No description')} (Source: {r.get('href', 'N/A')})" for r in results[:5]])
            return f"Current Web Search Results for '{query}':\n{formatted}"
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
