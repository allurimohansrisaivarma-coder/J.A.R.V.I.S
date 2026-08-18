"""Schemas for relational and vector databases."""

from datetime import UTC, datetime

import pyarrow as pa
from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Fact(Base):
    """SQLAlchemy model for structured facts."""
    __tablename__ = "facts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(String, nullable=False)
    source_context = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    is_active = Column(Boolean, default=True)

class Task(Base):
    """SQLAlchemy model for actionable tasks."""
    __tablename__ = "tasks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    description = Column(String, nullable=False)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    completed_at = Column(DateTime, nullable=True)

# LanceDB Schema for semantic memory
# all-MiniLM-L6-v2 generates 384-dimensional vectors
SemanticMemorySchema = pa.schema([
    pa.field("vector", pa.list_(pa.float32(), 384)),
    pa.field("content", pa.string()),
    pa.field("timestamp", pa.timestamp('us')),
    pa.field("type", pa.string()),  # e.g., "conversation_snippet", "extracted_fact"
])
