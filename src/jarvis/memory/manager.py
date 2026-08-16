"""High-level memory operations."""

import datetime
import structlog
from typing import List, Dict, Any

from jarvis.memory.database import get_sqlite_session, get_lancedb_table
from jarvis.memory.schema import Fact
from jarvis.memory.embeddings import Embedder

logger = structlog.get_logger(__name__)

class MemoryManager:
    """Core interface for storing and retrieving memories."""
    
    def __init__(self, embedder: Embedder | None = None):
        """Initialize the memory manager.
        
        Args:
            embedder: Embedder instance (will be created if None)
        """
        self.embedder = embedder or Embedder()
        self.table = get_lancedb_table()
        
    def store_fact(self, content: str, source_context: str = "") -> None:
        """Store a fact in both SQLite (for strict access) and LanceDB (for semantic search)."""
        logger.info("Storing memory fact", preview=content[:50])
        
        # 1. Store in SQLite
        db = get_sqlite_session()
        fact = Fact(content=content, source_context=source_context)
        db.add(fact)
        db.commit()
        db.refresh(fact)
        
        # 2. Store in LanceDB
        vector = self.embedder.embed_text(content)
        if not vector:
            logger.warning("Failed to embed fact, skipping LanceDB storage")
            return
            
        data = [{
            "vector": vector,
            "content": content,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "type": "extracted_fact"
        }]
        
        self.table.add(data)
        logger.debug("Fact stored in vector DB successfully")
        
    def semantic_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search memory semantically using vector similarity."""
        logger.debug("Searching memory", query=query)
        
        query_vector = self.embedder.embed_query(query)
        if not query_vector:
            return []
            
        # Perform vector search using LanceDB
        results = self.table.search(query_vector, vector_column_name="vector").limit(limit).to_list()
        
        # Format results slightly
        formatted = []
        for r in results:
            formatted.append({
                "content": r["content"],
                "distance": r["_distance"],
                "timestamp": r["timestamp"].isoformat() if hasattr(r["timestamp"], "isoformat") else r["timestamp"]
            })
            
        return formatted
