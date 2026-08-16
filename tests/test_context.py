"""Tests for the Context Engine."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from jarvis.context.engine import ContextEngine
from jarvis.context.base import ContextSource

class MockSource(ContextSource):
    def __init__(self, name, should_handle, result):
        self._name = name
        self.should_handle = should_handle
        self.result = result
        
    @property
    def name(self):
        return self._name
        
    async def can_handle(self, query):
        return self.should_handle
        
    async def gather_context(self, query):
        return self.result

@pytest.mark.asyncio
async def test_context_engine_aggregation():
    """Test that the engine aggregates active sources correctly."""
    engine = ContextEngine(memory_manager=None)
    
    # Replace default sources with mocks
    engine.sources = [
        MockSource("Source1", True, "Result 1"),
        MockSource("Source2", False, "Result 2"), # Should be skipped
        MockSource("Source3", True, "Result 3")
    ]
    
    prompt = await engine.build_context_prompt("test query")
    
    assert "System Information (World View)" in prompt
    assert "Context from Source1" in prompt
    assert "Result 1" in prompt
    assert "Context from Source3" in prompt
    assert "Result 3" in prompt
    assert "Source2" not in prompt

@pytest.mark.asyncio
async def test_context_engine_empty():
    """Test that the engine returns empty string if no sources have context."""
    engine = ContextEngine(memory_manager=None)
    
    engine.sources = [
        MockSource("Source1", False, ""),
        MockSource("Source2", True, "")
    ]
    
    prompt = await engine.build_context_prompt("test query")
    assert prompt == ""
