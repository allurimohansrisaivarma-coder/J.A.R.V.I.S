"""High-level memory operations."""

import datetime
from typing import Any

import structlog

from jarvis.memory.database import get_lancedb_table, get_sqlite_session
from jarvis.memory.embeddings import Embedder
from jarvis.memory.schema import Fact

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
            
        # 3. Deduplication Check
        # Check if an extremely similar memory already exists
        existing = self.table.search(vector, vector_column_name="vector").limit(1).to_list()
        if existing and existing[0]["_distance"] < 0.15:
            logger.info("Fact skipped (duplicate)", distance=existing[0]["_distance"])
            return
            
        data = [{
            "vector": vector,
            "content": content,
            "timestamp": datetime.datetime.now(datetime.UTC),
            "type": "extracted_fact"
        }]
        
        self.table.add(data)
        logger.debug("Fact stored in vector DB successfully")
        
    def semantic_search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
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
