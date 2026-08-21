import asyncio
import logging
import re
import sys
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import httpx
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("mcp_world_monitor")

mcp = FastMCP("World Monitor Service")

SEED_FEEDS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.cnbc.com/id/100727362/device/rss/rss.html",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
]

FINANCE_SEED_FEEDS = [
    "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "https://feeds.bloomberg.com/markets/news.rss",
    "https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
]


async def fetch_and_parse_feed(client, url):
    try:
        response = await client.get(url, headers={"User-Agent": "Friday-AI/1.0"}, timeout=5.0)
        if response.status_code != 200:
            return []

        root = ET.fromstring(response.content)
        hostname = urlparse(url).hostname or "Unknown"
        source_name = hostname.removeprefix("www.").split(".")[0].upper()

        feed_items = []
        items = root.findall(".//item")[:5]
        for item in items:
            title = item.findtext("title")
            description = item.findtext("description")
            link = item.findtext("link")

            if description:
                description = re.sub("<[^<]+?>", "", description).strip()

            feed_items.append(
                {
                    "source": source_name,
                    "title": title,
                    "summary": description[:200] + "..." if description else "",
                    "link": link,
                }
            )
        return feed_items
    except Exception as exc:
        logger.warning("Feed fetch failed for %s: %s", url, exc)
        return []


@mcp.tool()
async def open_world_monitor() -> str:
    """
    Open the JARVIS World Monitor Dashboard on the HUD.
    CRITICAL: Use this tool IMMEDIATELY when the user asks to "open the dashboard", "show my dashboard", "open world monitor", or wants to see their RAM usage, system stats, global news, or schedule.
    Do NOT use `launch_application` to open Chrome when the user asks for their dashboard.
    """
    return "WORLD_MONITOR_OPENED. Please tell the user you have opened the World Monitor dashboard on their screen."


@mcp.tool()
async def get_world_news() -> str:
    """
    Fetches the latest global headlines from major news outlets simultaneously.
    Use this when the user asks 'What's going on in the world?' or for recent events.
    """
    async with httpx.AsyncClient(follow_redirects=True, timeout=10) as client:
        tasks = [fetch_and_parse_feed(client, url) for url in SEED_FEEDS]
        results_of_lists = await asyncio.gather(*tasks)
        all_articles = [item for sublist in results_of_lists for item in sublist]

    if not all_articles:
        return "The global news grid is unresponsive, sir. I'm unable to pull headlines."

    report = ["### GLOBAL NEWS BRIEFING (LIVE)\n"]
    for entry in all_articles[:12]:
        report.append(f"**[{entry['source']}]** {entry['title']}")
        report.append(f"{entry['summary']}")
        report.append(f"Link: {entry['link']}\n")

    return "\n".join(report)


@mcp.tool()
async def get_world_finance_news() -> str:
    """
    Fetches the latest finance and market headlines from major financial outlets simultaneously.
    Use this when the user asks about finance news, market updates, or economic developments.
    """
    async with httpx.AsyncClient(follow_redirects=True, timeout=10) as client:
        tasks = [fetch_and_parse_feed(client, url) for url in FINANCE_SEED_FEEDS]
        results_of_lists = await asyncio.gather(*tasks)
        all_articles = [item for sublist in results_of_lists for item in sublist]

    if not all_articles:
        return "The financial feeds are unresponsive right now, sir. I can't pull market headlines."

    report = ["### FINANCE BRIEFING (LIVE)\n"]
    for entry in all_articles[:12]:
        report.append(f"**[{entry['source']}]** {entry['title']}")
        report.append(f"{entry['summary']}")
        report.append(f"Link: {entry['link']}\n")

    return "\n".join(report)


if __name__ == "__main__":
    logger.info("Starting World Monitor MCP Server...")
    mcp.run()
