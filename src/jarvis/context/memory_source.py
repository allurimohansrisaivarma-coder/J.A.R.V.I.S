"""Memory context source."""

import structlog
from jarvis.context.base import ContextSource
from jarvis.memory.manager import MemoryManager

logger = structlog.get_logger(__name__)

class MemoryContextSource(ContextSource):
    """Injects relevant facts from the memory system."""
    
    def __init__(self, memory_manager: MemoryManager | None = None):
        self.memory = memory_manager
        
    @property
    def name(self) -> str:
        return "Long-term Memory"
        
    async def can_handle(self, query: str) -> bool:
        """Memory is almost always relevant, but we only query if the memory system is online."""
        return self.memory is not None
        
    async def gather_context(self, query: str) -> str:
        if not self.memory:
            return ""
            
        try:
            results = self.memory.semantic_search(query, limit=5)
            # Filter results by a reasonable distance threshold (e.g. < 0.7 for cosine distance)
            relevant = [r for r in results if r["distance"] < 0.7]
            
            if not relevant:
                return ""
                
            formatted = "\n".join([f"- {r['content']}" for r in relevant])
            return f"Relevant memories about the user or past conversations:\n{formatted}"
        except Exception as e:
            logger.error("Failed to gather memory context", error=str(e))
            return ""
