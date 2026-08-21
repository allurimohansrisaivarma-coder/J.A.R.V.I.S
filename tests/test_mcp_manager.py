"""Tests for MCP schema conversion and lifecycle ordering."""

import pytest
from google.genai import types

from jarvis.tools.mcp_manager import MCPManager, _convert_schema_to_gemini


def test_json_schema_conversion_preserves_nested_types():
    schema = _convert_schema_to_gemini(
        {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "units": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["city"],
        }
    )

    assert schema.type == types.Type.OBJECT
    assert schema.properties["days"].type == types.Type.INTEGER
    assert schema.properties["units"].items.type == types.Type.STRING
    assert schema.required == ["city"]


@pytest.mark.asyncio
async def test_shutdown_closes_servers_in_reverse_start_order():
    closed: list[str] = []

    class FakeStack:
        def __init__(self, name: str):
            self.name = name

        async def aclose(self) -> None:
            closed.append(self.name)

    manager = MCPManager()
    manager._servers = {
        "first": {"stack": FakeStack("first")},
        "second": {"stack": FakeStack("second")},
        "third": {"stack": FakeStack("third")},
    }

    await manager.shutdown()

    assert closed == ["third", "second", "first"]
    assert manager._servers == {}
