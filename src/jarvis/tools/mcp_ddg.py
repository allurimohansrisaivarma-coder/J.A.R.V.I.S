import logging
import sys

from mcp.server.fastmcp import FastMCP

# Configure logging to stderr so it doesn't corrupt stdout JSON-RPC
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("mcp_ddg")

# Initialize FastMCP server
mcp = FastMCP("DuckDuckGo Web Search")


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the live internet for a given query and return web results.
    Use this tool to get up-to-date information, news, or general web facts.
    """
    from jarvis.context.web_evidence import search_evidence

    return search_evidence(query, max_results)


if __name__ == "__main__":
    logger.info("Starting DuckDuckGo MCP Server...")
    mcp.run()
