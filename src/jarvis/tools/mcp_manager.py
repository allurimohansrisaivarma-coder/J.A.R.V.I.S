"""MCP Manager for spawning and communicating with Model Context Protocol servers."""

import asyncio
from contextlib import AsyncExitStack
from typing import Any

import structlog
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = structlog.get_logger(__name__)


def _convert_schema_to_gemini(schema: dict) -> types.Schema:
    """Convert a JSON Schema dict into a Google genai types.Schema."""
    schema_type_mapping = {
        "string": types.Type.STRING,
        "integer": types.Type.INTEGER,
        "number": types.Type.NUMBER,
        "boolean": types.Type.BOOLEAN,
        "array": types.Type.ARRAY,
        "object": types.Type.OBJECT,
    }

    t = schema.get("type", "string").lower()
    genai_type = schema_type_mapping.get(t, types.Type.STRING)

    properties = {}
    if "properties" in schema:
        for k, v in schema["properties"].items():
            properties[k] = _convert_schema_to_gemini(v)

    items = None
    if "items" in schema:
        items = _convert_schema_to_gemini(schema["items"])

    return types.Schema(
        type=genai_type,
        description=schema.get("description", ""),
        properties=properties if properties else None,
        required=schema.get("required"),
        items=items,
    )


class MCPManager:
    """Manages connections to multiple MCP servers and exposes their tools."""

    def __init__(self):
        # Maps server_name -> (read_stream, write_stream, session)
        self._servers: dict[str, dict] = {}
        # Maps tool_name -> server_name
        self._tool_registry: dict[str, str] = {}

    async def start_server(self, name: str, command: str, args: list[str]) -> bool:
        """Start an MCP server subprocess."""
        logger.info("Starting MCP server", name=name, command=command, args=args)

        server_params = StdioServerParameters(command=command, args=args, env=None)

        stack = AsyncExitStack()
        try:
            read, write = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(read, write))

            await session.initialize()

            self._servers[name] = {"session": session, "stack": stack}

            # Fetch and register tools
            result = await asyncio.wait_for(session.list_tools(), timeout=10.0)
            for tool in result.tools:
                if tool.name in self._tool_registry:
                    logger.warning(
                        "Duplicate MCP tool ignored",
                        tool=tool.name,
                        existing_server=self._tool_registry[tool.name],
                        server=name,
                    )
                    continue
                logger.info("Registered MCP tool", tool=tool.name, server=name)
                self._tool_registry[tool.name] = name
            return True
        except Exception as e:
            logger.error("Failed to start MCP server", name=name, error=repr(e))
            await stack.aclose()
            return False

    async def get_gemini_tools(self, exclude: set[str] | None = None) -> types.Tool | None:
        """Get all registered tools formatted for Google Gemini SDK."""
        if not self._servers:
            return None

        declarations = []
        for server_name, server_data in self._servers.items():
            session = server_data["session"]
            try:
                result = await asyncio.wait_for(session.list_tools(), timeout=5.0)
                for tool in result.tools:
                    if exclude and tool.name in exclude:
                        continue
                    # Convert JSON Schema to genai types.Schema
                    params = None
                    if tool.inputSchema:
                        params = _convert_schema_to_gemini(tool.inputSchema)

                    declarations.append(
                        types.FunctionDeclaration(
                            name=tool.name, description=tool.description or "", parameters=params
                        )
                    )
            except Exception as e:
                logger.error("Failed to list tools for server", name=server_name, error=repr(e))

        if not declarations:
            return None

        return types.Tool(function_declarations=declarations)

    async def call_tool(self, name: str, args: dict[str, Any]) -> str:
        """Execute a tool on the appropriate MCP server."""
        server_name = self._tool_registry.get(name)
        if not server_name:
            return f"Error: Tool '{name}' is not registered."

        session = self._servers[server_name]["session"]
        logger.info("Calling MCP tool", tool=name, server=server_name)

        try:
            result = await session.call_tool(name, arguments=args)

            # Combine content blocks into a single string
            output = []
            for content in result.content:
                if content.type == "text":
                    output.append(content.text)
                else:
                    output.append(str(content))

            if result.isError:
                return f"Error from tool: {''.join(output)}"
            return "".join(output)

        except Exception as e:
            logger.error("Error executing MCP tool", tool=name, error=str(e))
            return f"Execution error: {e}"

    async def shutdown(self):
        """Close all MCP server connections."""
        # anyio cancel scopes are stack-like. Connections are opened by the
        # owning session task and must be closed in strict reverse order.
        for name, data in reversed(list(self._servers.items())):
            logger.info("Shutting down MCP server", name=name)
            try:
                await data["stack"].aclose()
            except Exception as e:
                if "generator didn't stop after athrow" not in str(e):
                    logger.warning("Error closing MCP server", name=name, error=str(e))
        self._servers.clear()
        self._tool_registry.clear()
