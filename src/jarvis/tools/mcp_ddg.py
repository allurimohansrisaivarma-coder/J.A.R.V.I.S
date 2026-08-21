import logging
import sys

from ddgs import DDGS
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
    logger.info(f"Executing web_search for query: {query}")
    try:
        ddgs = DDGS()
        results = ddgs.text(query, max_results=max_results)

        if not results:
            return "No web results found for the query."

        formatted_results = []
        for i, res in enumerate(results, 1):
            title = res.get("title", "No Title")
            href = res.get("href", "No URL")
            body = res.get("body", "No snippet available")
            formatted_results.append(f"Result {i}:\nTitle: {title}\nURL: {href}\nSnippet: {body}\n")

        return "\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Error during web search: {e}")
        return f"Error performing web search: {e}"


if __name__ == "__main__":
    logger.info("Starting DuckDuckGo MCP Server...")
    mcp.run()
