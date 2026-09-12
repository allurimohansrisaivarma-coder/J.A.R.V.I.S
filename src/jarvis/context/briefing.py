"""Daily news briefings rendered directly from fresh publisher RSS headlines."""

import asyncio
import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit, urlunsplit

import httpx


@dataclass(frozen=True)
class NewsTopic:
    name: str
    keywords: str
    feed: str
    publisher: str
    domains: tuple[str, ...]
    google_feed: str


TOPICS = (
    NewsTopic(
        "Cricket",
        r"\b(cricket|ashes|ranji)\b",
        "https://feeds.bbci.co.uk/sport/cricket/rss.xml",
        "BBC Sport",
        ("bbc.co.uk", "bbc.com"),
        "https://news.google.com/rss/search?q=cricket+when:1d&hl=en-GB&gl=GB&ceid=GB:en",
    ),
    NewsTopic(
        "Formula One",
        r"\b(f1|formula\s+(?:1|one)|grand prix)\b",
        "https://feeds.bbci.co.uk/sport/formula1/rss.xml",
        "BBC Sport",
        ("bbc.co.uk", "bbc.com"),
        "https://news.google.com/rss/search?q=Formula+1+when:1d&hl=en-GB&gl=GB&ceid=GB:en",
    ),
    NewsTopic(
        "AI",
        r"\b(ai|artificial intelligence)\b",
        "https://techcrunch.com/category/artificial-intelligence/feed/",
        "TechCrunch",
        ("techcrunch.com",),
        "https://news.google.com/rss/search?q=artificial+intelligence+when:1d&hl=en-US&gl=US&ceid=US:en",
    ),
    NewsTopic(
        "Software engineering",
        r"\b(software|engineering|developer|coding|programming)\b",
        "https://github.blog/feed/",
        "GitHub Blog",
        ("github.blog",),
        "https://news.google.com/rss/search?q=software+engineering+when:1d&hl=en-US&gl=US&ceid=US:en",
    ),
    NewsTopic(
        "World news",
        r"\b(world|politic\w*|global|impactful)\b",
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "BBC News",
        ("bbc.co.uk", "bbc.com"),
        "https://news.google.com/rss/search?q=world+news+when:1d&hl=en-US&gl=US&ceid=US:en",
    ),
)


def is_news_briefing(query: str) -> bool:
    # Speech recognition commonly separates or hyphenates "debrief".
    normalized = re.sub(r"\bde[\s-]+brief\b", "debrief", query, flags=re.IGNORECASE)
    known_topic_news = bool(
        re.search(r"\b(?:news|headlines?|updates?)\b", normalized, re.IGNORECASE)
        and any(re.search(topic.keywords, normalized, re.IGNORECASE) for topic in TOPICS)
    )
    if not known_topic_news and not re.search(
        r"\b(?:(?:(?:daily|morning|evening)(?:\s+news)?|news|sports)\s+(?:debrief|briefing|brief|digest|roundup)|(?:debrief|briefing|roundup)\s+(?:of\s+)?(?:the\s+)?news)\b",
        normalized,
        re.IGNORECASE,
    ):
        return False
    # A meeting/project debrief is private context, not a public-news request.
    private = re.search(
        r"\b(meeting|project|calendar|inbox|workday|work day|files?|documents?)\b",
        query,
        re.IGNORECASE,
    )
    public = re.search(r"\bnews\b", query, re.IGNORECASE) or any(
        re.search(t.keywords, query, re.IGNORECASE) for t in TOPICS
    )
    return not private or bool(public)


def selected_topics(query: str) -> tuple[NewsTopic, ...]:
    explicit = tuple(topic for topic in TOPICS if re.search(topic.keywords, query, re.IGNORECASE))
    return explicit or TOPICS


class _FeedTreeBuilder(ET.TreeBuilder):
    def doctype(self, name, pubid, system):
        raise ValueError("Feed document types are not supported")


def parse_headlines(
    data: bytes,
    topic: NewsTopic,
    now: datetime,
    *,
    google_news: bool = False,
) -> list[dict]:
    if len(data) > 2_000_000:
        return []
    try:
        root = ET.fromstring(data, parser=ET.XMLParser(target=_FeedTreeBuilder()))
    except (ET.ParseError, ValueError):
        return []
    rows = []
    seen: set[str] = set()
    for item in root.findall(".//item")[:100]:
        try:
            published = parsedate_to_datetime(item.findtext("pubDate", ""))
            if published.tzinfo is None or not timedelta(0) <= now - published <= timedelta(
                hours=24
            ):
                continue
            title = html.unescape(html.unescape(item.findtext("title", "")))
            title = re.sub(r"<[^>]+>", "", title)
            title = " ".join(title.split())
            parsed = urlsplit(html.unescape(item.findtext("link", "")))
            host = (parsed.hostname or "").casefold()
            allowed_domains = ("news.google.com",) if google_news else topic.domains
            if (
                parsed.scheme != "https"
                or parsed.username
                or parsed.password
                or not any(
                    host == domain or host.endswith("." + domain) for domain in allowed_domains
                )
            ):
                continue
            if not title or len(title) > 350:
                continue
            url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
            title_key = re.sub(r"\W+", "", title).casefold()
            if url in seen or title_key in seen:
                continue
            seen.update((url, title_key))
            publisher = topic.publisher
            if google_news:
                publisher = html.unescape(item.findtext("source", "Google News")).strip()
                publisher = " ".join(re.sub(r"[\[\]*`\\]", "", publisher).split())[:80]
                publisher = publisher or "Google News"
            rows.append(
                {"title": title, "url": url, "published": published, "publisher": publisher}
            )
        except (TypeError, ValueError, OverflowError):
            continue
    return sorted(rows, key=lambda row: row["published"], reverse=True)


async def fetch_headlines(client: httpx.AsyncClient, topic: NewsTopic, now: datetime) -> list[dict]:
    primary = await _fetch_feed(client, topic.feed, topic, now)
    if len(primary) >= 2:
        return primary
    fallback = await _fetch_feed(client, topic.google_feed, topic, now, google_news=True)
    combined = primary + fallback
    seen_titles = set()
    unique = []
    for row in sorted(combined, key=lambda item: item["published"], reverse=True):
        key = re.sub(r"\W+", "", row["title"]).casefold()
        if key not in seen_titles:
            seen_titles.add(key)
            unique.append(row)
    return unique


async def _fetch_feed(
    client: httpx.AsyncClient,
    feed: str,
    topic: NewsTopic,
    now: datetime,
    *,
    google_news: bool = False,
) -> list[dict]:
    try:
        async with client.stream("GET", feed) as response:
            response.raise_for_status()
            chunks = []
            size = 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > 2_000_000:
                    return []
                chunks.append(chunk)
        return parse_headlines(b"".join(chunks), topic, now, google_news=google_news)
    except (httpx.HTTPError, ValueError):
        return []


async def build_daily_briefing(query: str, *, now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    topics = selected_topics(query)
    count = 10 if re.search(r"\b(?:10|ten)\b", query, re.IGNORECASE) else min(10, 2 * len(topics))
    per_topic = max(1, count // len(topics))
    async with httpx.AsyncClient(
        timeout=4, follow_redirects=False, headers={"User-Agent": "JARVIS news reader"}
    ) as client:
        results = await asyncio.gather(
            *(asyncio.wait_for(fetch_headlines(client, topic, now), timeout=6) for topic in topics),
            return_exceptions=True,
        )
    local = now.astimezone()
    output = [
        f"**Daily Debrief — {local:%d %b %Y}**",
        f"Publisher headlines from the past 24 hours. Checked {now:%H:%M UTC}. Each date below is the publication time.",
    ]
    for topic, result in zip(topics, results):
        output.append(f"**{topic.name}**")
        if isinstance(result, BaseException) or not result:
            output.append(
                "No fresh, dated headlines could be retrieved from this feed. I won't fill the gap with older news."
            )
            continue
        for row in result[:per_topic]:
            # Literal publisher titles only: no model-generated scores, standings,
            # schedules or invented summaries; strip Markdown control characters.
            title = re.sub(r"[\[\]*`\\]", "", row["title"])
            output.append(
                f"- [{title}]({row['url']}) — {row.get('publisher', topic.publisher)}, {row['published'].astimezone(UTC):%d %b %H:%M UTC}."
            )
    return "\n\n".join(output)
