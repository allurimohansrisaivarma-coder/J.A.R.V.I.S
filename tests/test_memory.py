"""Tests for the memory system."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from jarvis.memory.database import init_db, get_sqlite_session, get_lancedb_table
from jarvis.memory.schema import Fact
from jarvis.memory.manager import MemoryManager
from jarvis.memory.embeddings import Embedder

@pytest.fixture
def mock_embedder():
    embedder = MagicMock(spec=Embedder)
    # Return a 384-dimensional mock vector
    embedder.embed_text.return_value = [0.1] * 384
    embedder.embed_query.return_value = [0.1] * 384
    return embedder

@pytest.fixture
def memory_db(tmp_path):
    """Initializes the database in a temp directory for tests."""
    init_db(tmp_path)
    yield
    # No need to explicitly teardown for sqlite/lancedb file locks on posix, 
    # but windows might complain on tempdir cleanup if lancedb holds a lock. 
    # That's fine for simple unit tests.

def test_store_and_recall_fact(memory_db, mock_embedder):
    """Test that we can store a fact in SQLite and LanceDB, and recall it."""
    manager = MemoryManager(embedder=mock_embedder)
    
    manager.store_fact("The user likes coffee.", "test_context")
    
    # 1. Check SQLite
    db = get_sqlite_session()
    facts = db.query(Fact).all()
    assert len(facts) == 1
    assert facts[0].content == "The user likes coffee."
    assert facts[0].source_context == "test_context"
    
    # 2. Check LanceDB
    table = get_lancedb_table()
    assert table.count_rows() == 1
    
    # 3. Check Semantic Search
    results = manager.semantic_search("What does the user like?", limit=5)
    assert len(results) == 1
    assert results[0]["content"] == "The user likes coffee."
