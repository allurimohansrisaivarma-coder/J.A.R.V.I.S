"""High-level memory operations."""

import datetime
import fnmatch
from typing import Any

import structlog

from jarvis.memory.database import get_lancedb_table, get_sqlite_session
from jarvis.memory.embeddings import Embedder
from jarvis.memory.schema import Fact

logger = structlog.get_logger(__name__)


class MemoryManager:
    """Core interface for storing and retrieving memories."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        exclude_patterns: list[str] | None = None,
    ):
        """Initialize the memory manager.

        Args:
            embedder: Embedder instance (will be created if None)
        """
        self.embedder = embedder or Embedder()
        self.table = get_lancedb_table()
        self.exclude_patterns = exclude_patterns or []

    def _is_excluded(self, content: str) -> bool:
        lowered = content.lower()
        return any(
            pattern.lower() in lowered or fnmatch.fnmatch(lowered, pattern.lower())
            for pattern in self.exclude_patterns
        )

    def store_fact(self, content: str, source_context: str = "") -> None:
        """Store a fact in both SQLite (for strict access) and LanceDB (for semantic search)."""
        content = content.strip()
        if not content or self._is_excluded(content):
            logger.info("Memory fact skipped by exclusion policy")
            return

        logger.info("Storing memory fact", content_length=len(content))

        vector = self.embedder.embed_text(content)
        if not vector:
            raise RuntimeError("Failed to generate an embedding for the memory")

        # Check semantic duplicates before writing either database.
        existing = self.table.search(vector, vector_column_name="vector").limit(1).to_list()
        if existing and existing[0]["_distance"] < 0.15:
            logger.info("Fact skipped (duplicate)", distance=existing[0]["_distance"])
            return

        db = get_sqlite_session()
        fact = Fact(content=content, source_context=source_context)
        try:
            if db.query(Fact).filter(Fact.content == content, Fact.is_active.is_(True)).first():
                logger.info("Fact skipped (exact duplicate)")
                return
            db.add(fact)
            db.commit()
            db.refresh(fact)

            data = [
                {
                    "vector": vector,
                    "content": content,
                    "timestamp": datetime.datetime.now(datetime.UTC),
                    "type": "extracted_fact",
                }
            ]
            self.table.add(data)
        except Exception:
            db.rollback()
            if fact.id is not None:
                db.delete(fact)
                db.commit()
            raise
        finally:
            db.close()

        logger.debug("Fact stored in vector DB successfully")

    def semantic_search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search memory semantically using vector similarity."""
        logger.debug("Searching memory", query=query)

        query_vector = self.embedder.embed_query(query)
        if not query_vector:
            return []

        # Perform vector search using LanceDB
        results = (
            self.table.search(query_vector, vector_column_name="vector").limit(limit).to_list()
        )

        # Format results slightly
        formatted = []
        for r in results:
            formatted.append(
                {
                    "content": r["content"],
                    "distance": r["_distance"],
                    "timestamp": r["timestamp"].isoformat()
                    if hasattr(r["timestamp"], "isoformat")
                    else r["timestamp"],
                }
            )

        return formatted

    def delete_memory(self, query: str, dry_run: bool = True) -> str:
        """Delete a memory matching the query. If dry_run is True, returns the matched memory without deleting."""
        logger.info("Attempting memory deletion", query=query, dry_run=dry_run)

        query_vector = self.embedder.embed_query(query)
        if not query_vector:
            return "Failed to process query for deletion."

        results = self.table.search(query_vector, vector_column_name="vector").limit(1).to_list()

        if not results or results[0]["_distance"] > 0.4:
            return "No close matching memory found to delete. Please be more specific."

        target_content = results[0]["content"]

        if dry_run:
            return (
                f"Found matching memory: '{target_content}'. Set dry_run=False to confirm deletion."
            )

        # Actually delete
        db = None
        try:
            # Delete from SQLite
            db = get_sqlite_session()
            facts = db.query(Fact).filter(Fact.content == target_content).all()
            for fact in facts:
                db.delete(fact)
            db.commit()

            # Delete from LanceDB
            # LanceDB where clause uses SQL syntax. Escaping single quotes is done by doubling them.
            safe_content = target_content.replace("'", "''")
            self.table.delete(f"content = '{safe_content}'")

            logger.info("Memory deleted successfully", content=target_content)
            return f"Successfully deleted memory: '{target_content}'"
        except Exception as e:
            logger.error("Failed to delete memory", error=str(e))
            return f"Failed to delete memory due to database error: {e}"
        finally:
            if db is not None:
                db.close()
