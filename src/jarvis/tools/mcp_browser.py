import asyncio
import logging
import sys
from urllib.parse import quote_plus, urlparse

from mcp.server.fastmcp import FastMCP
from playwright.async_api import Browser, Page, Playwright, async_playwright

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("mcp_browser")

mcp = FastMCP("Chrome Browser Control")

_playwright: Playwright | None = None
_browser: Browser | None = None
_page: Page | None = None


async def get_page() -> Page:
    global _playwright, _browser, _page
    if not _playwright:
        _playwright = await async_playwright().start()
    if _browser and not _browser.is_connected():
        _browser = None
        _page = None
    if not _browser:
        try:
            _browser = await _playwright.chromium.launch(channel="chrome", headless=False)
        except Exception:
            try:
                _browser = await _playwright.chromium.launch(channel="msedge", headless=False)
            except Exception:
                _browser = await _playwright.chromium.launch(headless=False)
    if not _page or _page.is_closed():
        _page = await _browser.new_page()
        _page.set_default_timeout(10000)
        _page.set_default_navigation_timeout(15000)
    return _page


async def shutdown():
    global _playwright, _browser, _page
    if _playwright:
        await _playwright.stop()
    _playwright = _browser = _page = None


@mcp.tool()
async def browser_navigate(url: str) -> str:
    """
    Navigate the browser to a specific URL.
    """
    try:
        page = await get_page()
        if not urlparse(url).scheme:
            url = "https://" + url
        if urlparse(url).scheme not in {"http", "https"}:
            return "Only http and https URLs can be opened."
        await page.goto(url, wait_until="domcontentloaded")
        return f"Navigated to {page.url}"
    except Exception as e:
        return f"Failed to navigate: {e}"


@mcp.tool()
async def browser_search(query: str) -> str:
    """
    Perform a Google search in the browser.
    """
    try:
        page = await get_page()
        await page.goto(
            f"https://www.google.com/search?q={quote_plus(query)}",
            wait_until="domcontentloaded",
        )
        return f"Search completed for '{query}'. Current URL: {page.url}"
    except Exception as e:
        return f"Failed to search: {e}"


@mcp.tool()
async def browser_read_page() -> str:
    """
    Extract the readable text content from the current active browser tab.
    """
    try:
        page = await get_page()
        text = await page.evaluate("document.body.innerText")
        if not text:
            return "Page is empty or could not be read."
        # Truncate to avoid context window explosion
        return text[:15000]
    except Exception as e:
        return f"Failed to read page: {e}"


@mcp.tool()
async def browser_click(selector: str) -> str:
    """
    Click on an element on the page using a CSS selector (e.g. 'a', 'button#submit').
    """
    try:
        page = await get_page()
        await page.click(selector)
        # Give it a tiny bit of time to start any navigations
        await asyncio.sleep(1)
        return f"Clicked '{selector}'. Current URL: {page.url}"
    except Exception as e:
        return f"Failed to click '{selector}': {e}"


if __name__ == "__main__":
    logger.info("Starting Chrome Browser Control MCP Server...")
    mcp.run()
