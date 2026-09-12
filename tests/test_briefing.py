"""A debrief must never fabricate current events from conversation history."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from jarvis.context.briefing import (
    TOPICS,
    build_daily_briefing,
    is_news_briefing,
    parse_headlines,
    selected_topics,
)
from jarvis.core.conversation import ConversationManager
from jarvis.core.session import SessionManager

NOW = datetime(2026, 9, 12, 2, tzinfo=UTC)


def feed(*items):
    return (
        "<rss><channel>"
        + "".join(
            f"<item><title>{title}</title><link>{url}</link><pubDate>{date}</pubDate></item>"
            for title, url, date in items
        )
        + "</channel></rss>"
    ).encode()


def test_only_fresh_dated_publisher_headlines_are_accepted():
    data = feed(
        (
            "Fresh &amp; accurate &#x20; headline",
            "https://www.bbc.co.uk/sport/one?at_medium=RSS",
            "Fri, 11 Sep 2026 18:00:00 GMT",
        ),
        ("Old Ashes result", "https://www.bbc.co.uk/sport/two", "Wed, 09 Sep 2026 18:00:00 GMT"),
        ("Future race", "https://www.bbc.co.uk/sport/three", "Sun, 13 Sep 2026 18:00:00 GMT"),
        ("Undated claim", "https://www.bbc.co.uk/sport/four", ""),
        ("Untrusted source", "https://untrusted.example/sport", "Fri, 11 Sep 2026 18:00:00 GMT"),
    )
    rows = parse_headlines(data, TOPICS[0], NOW)
    assert len(rows) == 1
    assert rows[0]["title"] == "Fresh & accurate headline"
    assert rows[0]["url"] == "https://www.bbc.co.uk/sport/one"


def test_code_examples_in_feeds_are_allowed_but_real_dtds_are_rejected():
    data = feed(("New article", "https://github.blog/article", "Fri, 11 Sep 2026 18:00:00 GMT"))
    data = data.replace(
        b"</item>", b"<description><![CDATA[Example: <!DOCTYPE html>]]></description></item>"
    )
    assert len(parse_headlines(data, TOPICS[3], NOW)) == 1
    assert parse_headlines(b'<!DOCTYPE rss [<!ENTITY x "expansion">]>' + data, TOPICS[3], NOW) == []


def test_google_news_fallback_keeps_dated_headline_and_named_publisher():
    data = (
        b"<rss><channel><item><title>Verified event - Reuters</title>"
        b"<link>https://news.google.com/rss/articles/example</link>"
        b"<pubDate>Fri, 11 Sep 2026 18:00:00 GMT</pubDate>"
        b"<source url='https://reuters.com'>Reuters</source></item></channel></rss>"
    )

    rows = parse_headlines(data, TOPICS[4], NOW, google_news=True)

    assert len(rows) == 1
    assert rows[0]["publisher"] == "Reuters"
    assert rows[0]["url"].startswith("https://news.google.com/")


@pytest.mark.parametrize(
    "query",
    [
        "Give me my daily debrief.",
        "give me the daily de brief",
        "Give me a daily de-brief",
        "i didnt get the ai news",
        "you missed AI",
        "repeat Formula One",
        "Read the AI headlines again",
        "Daily briefing",
        "Morning news brief",
        "Daily debrief and do you remember where I live?",
    ],
)
def test_briefing_aliases(query):
    assert is_news_briefing(query)


@pytest.mark.parametrize(
    "query", ["Debrief my meeting", "Daily debrief of my project", "Daily briefing for my inbox"]
)
def test_private_debrief_is_not_public_news(query):
    assert not is_news_briefing(query)


def test_explicit_topic_request_does_not_mix_unrelated_news():
    assert [t.name for t in selected_topics("Top ten AI and Formula 1 daily debrief")] == [
        "Formula One",
        "AI",
    ]


@pytest.mark.asyncio
async def test_digest_uses_only_feed_titles_with_dates_and_omits_empty_categories(
    monkeypatch,
):
    async def fetch(client, topic, now):
        return (
            [
                {
                    "title": "A real publisher headline",
                    "url": "https://www.bbc.co.uk/sport/article",
                    "published": NOW,
                }
            ]
            if topic.name == "Formula One"
            else []
        )

    monkeypatch.setattr("jarvis.context.briefing.fetch_headlines", fetch)
    answer = await build_daily_briefing("Daily debrief", now=NOW)
    assert "[A real publisher headline](https://www.bbc.co.uk/sport/article)" in answer
    assert "12 Sep 02:00 UTC" in answer
    assert "**Formula One**" in answer
    assert "**Cricket**" not in answer
    assert "**AI**" not in answer
    assert "No fresh, dated" not in answer
    assert "Hamilton" not in answer and "Ashes" not in answer


@pytest.mark.asyncio
async def test_feed_failure_and_old_results_do_not_become_news(monkeypatch):
    async def fetch(client, topic, now):
        raise httpx.ConnectError("unavailable")

    monkeypatch.setattr("jarvis.context.briefing.fetch_headlines", fetch)
    answer = await build_daily_briefing("Daily debrief", now=NOW)
    assert answer.count("No fresh, dated publisher headlines") == 1
    assert not any(f"**{topic.name}**" in answer for topic in TOPICS)


@pytest.mark.asyncio
async def test_topic_followup_contains_only_requested_feed_and_no_old_context(monkeypatch):
    async def fetch(client, topic, now):
        assert topic.name == "AI"
        return [
            {
                "title": "Verified AI publisher headline",
                "url": "https://techcrunch.com/verified-ai-headline",
                "published": NOW,
                "publisher": "TechCrunch",
            }
        ]

    monkeypatch.setattr("jarvis.context.briefing.fetch_headlines", fetch)
    answer = await build_daily_briefing("you missed AI", now=NOW)

    assert "Verified AI publisher headline" in answer
    assert "**AI**" in answer
    assert "**Cricket**" not in answer
    assert "**Formula One**" not in answer
    assert "**Software engineering**" not in answer
    assert "**World news**" not in answer
    assert "Sunset" not in answer
    assert "BITS" not in answer
    assert "mohan cv" not in answer


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_reported_daily_debrief_never_uses_llm_news_generation(monkeypatch, streaming):
    session = SessionManager()
    session._initialized = True
    session.router = MagicMock()
    session.router.providers = {}
    session.conversation = ConversationManager()
    session.conversation.add_assistant_message(
        "Old invented debrief: Hamilton leads, Bahrain on September 18, England lead the Ashes."
    )
    monkeypatch.setattr(
        "jarvis.core.session.build_daily_briefing",
        AsyncMock(return_value="Fresh dated publisher headlines"),
    )
    monkeypatch.setattr(session, "_append_transcript", MagicMock())
    spoken_variant = "give me the daily de brief"
    if streaming:
        answer = "".join(
            [
                chunk
                async for chunk in session.process_input_stream(spoken_variant)
                if isinstance(chunk, str)
            ]
        )
    else:
        answer = await session.process_input(spoken_variant)
    assert answer == "Fresh dated publisher headlines"
    session.router.generate_with_fallback.assert_not_called()
    session.router.route_stream.assert_not_called()


@pytest.mark.asyncio
async def test_compound_personal_question_is_separate_and_old_assistant_claims_are_excluded(
    monkeypatch,
):
    session = SessionManager()
    session.router = MagicMock()
    session.router.providers = {"test": True}
    session.router.generate_with_fallback = AsyncMock(
        return_value=SimpleNamespace(content="You live in the saved test city.")
    )
    session.conversation = ConversationManager()
    session.conversation.add_user_message("I live in the saved test city.")
    session.conversation.add_assistant_message("Invented cricket score")
    query = "Give me my daily debrief and do you remember where I live?"
    session.conversation.add_user_message(query)
    monkeypatch.setattr(
        "jarvis.core.session.build_daily_briefing", AsyncMock(return_value="Fresh headline")
    )
    answer = await session._try_daily_briefing(query)
    assert answer == "Fresh headline\n\nYou live in the saved test city."
    messages = session.router.generate_with_fallback.call_args.args[0]
    assert messages[-1].content == "do you remember where I live?"
    assert not any(m.role == "assistant" for m in messages)
    assert "Invented cricket score" not in str(messages)


@pytest.mark.asyncio
async def test_continue_after_spoken_briefing_cannot_invent_more_news(monkeypatch):
    session = SessionManager()
    session._initialized = True
    session.router = MagicMock()
    session.router.providers = {}
    session.conversation = ConversationManager()
    session.conversation.add_user_message("give me the daily de brief")
    session.conversation.add_assistant_message("Old invented news")
    monkeypatch.setattr(session, "_append_transcript", MagicMock())

    answer = await session.process_input("continue")

    assert "complete set of fresh, dated headlines" in answer
    session.router.generate_with_fallback.assert_not_called()
    session.router.route_stream.assert_not_called()


@pytest.mark.asyncio
async def test_missed_topic_followup_replays_verified_feed_without_model(monkeypatch):
    session = SessionManager()
    session._initialized = True
    session.router = MagicMock()
    session.router.providers = {}
    session.conversation = ConversationManager()
    session.conversation.add_assistant_message("An older unverified AI list")
    monkeypatch.setattr(
        "jarvis.core.session.build_daily_briefing",
        AsyncMock(return_value="Two verified AI publisher headlines"),
    )
    monkeypatch.setattr(session, "_append_transcript", MagicMock())

    answer = await session.process_input("i didnt get the ai news")

    assert answer == "Two verified AI publisher headlines"
    session.router.generate_with_fallback.assert_not_called()
    session.router.route_stream.assert_not_called()
