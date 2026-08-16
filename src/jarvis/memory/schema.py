"""Schemas for relational and vector databases."""

from datetime import datetime, timezone
import pyarrow as pa
from sqlalchemy import Column, String, Integer, DateTime, Boolean
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Fact(Base):
    """SQLAlchemy model for structured facts."""
    __tablename__ = "facts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(String, nullable=False)
    source_context = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)

class Task(Base):
    """SQLAlchemy model for actionable tasks."""
    __tablename__ = "tasks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    description = Column(String, nullable=False)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)

# LanceDB Schema for semantic memory
# gemini-embedding-2 generates 3072-dimensional vectors
SemanticMemorySchema = pa.schema([
    pa.field("vector", pa.list_(pa.float32(), 3072)),
    pa.field("content", pa.string()),
    pa.field("timestamp", pa.timestamp('us')),
    pa.field("type", pa.string()),  # e.g., "conversation_snippet", "extracted_fact"
])
