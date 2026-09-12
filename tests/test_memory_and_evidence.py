"""Regression coverage for persistent recall, grounded search, and voice identity."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.context.web_evidence import WEB_UNAVAILABLE, format_evidence, public_url
from jarvis.context.web_source import WebContextSource
from jarvis.core.conversation import ConversationManager
from jarvis.core.session import SessionManager
from jarvis.voice.tts import TTSProvider


def test_history_survives_restart_and_recalls_newer_corrections(tmp_path):
    path = tmp_path / "history.sqlite3"
    original = ConversationManager(storage_path=path)
    original.add_user_message("My project codename is Orion.")
    original.add_assistant_message("An invented codename is Polaris.")
    original.add_user_message("Correction: my project codename is Nebula now.")
    restored = ConversationManager(storage_path=path)
    assert restored.get_active().id == original.get_active().id
    assert len(restored.get_active().messages) == 4
    restored.new_conversation()
    restored.add_user_message("Remember my project codename?")
    context = "\n".join(str(m.content) for m in restored.get_context_messages())
    assert "Orion" in context and "Nebula" in context
    assert context.index("Orion") < context.index("Nebula")
    assert "Polaris" not in context


def test_import_skips_broken_rows_and_clear_does_not_reimport(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        "[]\nnull\nbroken\n"
        + json.dumps({"role": "user", "text": "I enjoy astronomy"})
        + "\n"
        + json.dumps({"role": "jarvis", "text": "partial", "status": "interrupted"})
    )
    path = tmp_path / "history.sqlite3"
    memory = ConversationManager(storage_path=path, transcript=transcript)
    assert len(memory.get_active().messages) == 2
    memory.clear_history()
    restarted = ConversationManager(storage_path=path, transcript=transcript)
    restarted.add_user_message("Remember astronomy?")
    assert "I enjoy astronomy" not in str(restarted.get_context_messages())


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "Error from tool: unavailable",
        "null",
        "[]",
        '{"status":"ok","results":[]}',
        '{"status":"ok","results":[{"url":"https://example.com"}]}',
    ],
)
def test_empty_or_failed_search_never_becomes_evidence(raw):
    assert format_evidence(raw) == WEB_UNAVAILABLE


def test_search_citations_keep_dates_and_snippet_limitations():
    evidence = format_evidence(
        json.dumps(
            {
                "status": "ok",
                "retrieved_at": "2026-09-11",
                "results": [
                    {
                        "id": 1,
                        "url": "https://example.com/report",
                        "title": "Report",
                        "published": "2026-08-01",
                        "snippet": "Only this is supported.",
                    }
                ],
            }
        )
    )
    assert "https://example.com/report" in evidence
    assert "Publication date: 2026-08-01" in evidence
    assert "Search snippet only; page not verified" in evidence
    assert "NOT the publication date" in evidence


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1",
        "http://[::1]",
        "http://169.254.169.254",
        "file:///secret",
        "https://user:password@example.com",
    ],
)
def test_page_reader_rejects_private_addresses(url):
    assert not public_url(url)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    ["Do you know my name?", "Explain hardware", "Find my conversation", "Search my documents"],
)
async def test_private_and_unrelated_queries_do_not_trigger_search(query):
    assert not await WebContextSource().can_handle(query)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_failed_live_retrieval_never_reaches_generation(monkeypatch, streaming):
    session = SessionManager()
    session._initialized = True
    session.conversation = ConversationManager()
    session.router = MagicMock()
    session.router.providers = {"test": True}
    session.context_engine = SimpleNamespace(
        build_context_prompt=AsyncMock(return_value=(WEB_UNAVAILABLE, []))
    )
    monkeypatch.setattr(session, "_append_transcript", MagicMock())
    if streaming:
        answer = "".join(
            [
                chunk
                async for chunk in session.process_input_stream("Latest F1 score")
                if isinstance(chunk, str)
            ]
        )
    else:
        answer = await session.process_input("Latest F1 score")
    assert "won't guess" in answer
    assert not session.router.generate_with_fallback.called
    assert not session.router.route_stream.called


def test_short_followup_keeps_live_topic_but_not_stale_topic():
    session = SessionManager()
    session.conversation = ConversationManager()
    session.conversation.add_user_message("Weather in Dubai today")
    session.conversation.add_user_message("What about tomorrow?")
    assert "Dubai" in session._resolve_context_query("What about tomorrow?")
    session.conversation.add_user_message("Explain recursion")
    session.conversation.add_user_message("Yes")
    assert session._resolve_context_query("Yes") == "Yes"


@pytest.mark.asyncio
async def test_failed_screen_capture_is_not_misreported_as_web_failure():
    assert not await SessionManager._needs_web_evidence("What is on my screen now?")
    assert not await SessionManager._needs_web_evidence("What time is it now?")


@pytest.mark.asyncio
async def test_tts_retry_uses_identical_voice_and_settings(monkeypatch):
    tts = TTSProvider()
    save = AsyncMock(side_effect=[ConnectionError("temporary"), None])
    communicate = MagicMock(return_value=SimpleNamespace(save=save))
    monkeypatch.setattr("jarvis.voice.tts.edge_tts.Communicate", communicate)
    await tts._save_voice("Test", "unused.mp3", "en-GB-RyanNeural", "+20%", "-5Hz")
    assert communicate.call_count == 2
    assert communicate.call_args_list[0] == communicate.call_args_list[1]
    assert not hasattr(tts, "_speak_offline")


@pytest.mark.asyncio
async def test_tts_survives_three_transient_failures_without_switching_voice(monkeypatch):
    tts = TTSProvider()
    save = AsyncMock(
        side_effect=[
            ConnectionError("temporary 1"),
            TimeoutError("temporary 2"),
            ConnectionError("temporary 3"),
            None,
        ]
    )
    communicate = MagicMock(return_value=SimpleNamespace(save=save))
    monkeypatch.setattr("jarvis.voice.tts.edge_tts.Communicate", communicate)
    monkeypatch.setattr("jarvis.voice.tts.asyncio.sleep", AsyncMock())

    await tts._save_voice("Test", "unused.mp3", "en-GB-RyanNeural", "+20%", "-5Hz")

    assert communicate.call_count == 4
    assert all(call == communicate.call_args_list[0] for call in communicate.call_args_list)


@pytest.mark.asyncio
async def test_voice_is_frozen_for_entire_reply_and_failures_keep_text(monkeypatch):
    import pygame

    tts = TTSProvider()
    tts._use_pygame = True
    notices = []
    tts.on_unavailable = notices.append
    saved = []

    async def save(text, path, voice, rate, pitch):
        saved.append((voice, rate, pitch))
        if len(saved) > 1:
            raise ConnectionError("outage")

    monkeypatch.setattr(tts, "_save_voice", save)
    monkeypatch.setattr(pygame.mixer, "music", MagicMock(get_busy=MagicMock(return_value=False)))

    async def chunks():
        yield "This first phrase is long enough to be spoken. "
        tts.voice = "en-US-AriaNeural"
        tts.rate = "+5%"
        yield "This second phrase must keep the same selected voice. "
        yield "This third phrase remains available in the conversation."

    phrases = []
    await tts.speak_stream(chunks(), phrases.append)
    assert len(saved) == 3
    assert all(settings == ("en-GB-RyanNeural", "+20%", "-5Hz") for settings in saved)
    assert len(notices) == 1
    assert "first phrase" in phrases[0]
    assert "third phrase" in "".join(phrases)
    assert not tts.is_speaking


@pytest.mark.parametrize(
    "link",
    [
        "https://www.formula1.com/en/results/2026/races",
        "www.formula1.com/en/results/2026/races",
        "formula1.com/en/results/2026/races",
        "[Official Formula 1 race results](https://www.formula1.com/en/results/2026/races)",
    ],
)
@pytest.mark.asyncio
async def test_streamed_urls_are_visible_but_never_spoken(monkeypatch, link):
    import pygame

    tts = TTSProvider()
    tts._use_pygame = True
    spoken = []
    notices = []
    tts.on_unavailable = notices.append

    async def save(text, path, voice, rate, pitch):
        spoken.append(text)

    monkeypatch.setattr(tts, "_save_voice", save)
    monkeypatch.setattr(pygame.mixer, "music", MagicMock(get_busy=MagicMock(return_value=False)))
    text = (
        "The reported race result is available from the official source. Source: "
        + link
        + " 【1】 That is the verified result."
    )

    async def chunks():
        for offset in range(0, len(text), 3):
            yield text[offset : offset + 3]

    visible = []
    await tts.speak_stream(chunks(), visible.append)
    audio_text = " ".join(spoken)
    assert "formula1" not in audio_text.lower()
    assert "http" not in audio_text and "2026/races" not in audio_text
    assert "【" not in audio_text
    assert link in "".join(visible)
    assert "verified result" in audio_text
    assert not notices


def test_windows_file_paths_are_visible_in_chat_but_not_spoken():
    provider = object.__new__(TTSProvider)
    reply = "The file is located at `C:\\Users\\person\\OneDrive\\Desktop\\Mohan CV.pdf`."

    spoken = provider._clean_text(reply)

    assert "C:" not in spoken
    assert "OneDrive" not in spoken
    assert "file location shown in chat" in spoken


@pytest.mark.parametrize("formatting_only", ["-", "—", "***", "[1]", "  ...  "])
def test_formatting_only_chunks_are_not_sent_to_voice(formatting_only):
    provider = object.__new__(TTSProvider)
    assert provider._clean_text(formatting_only) == ""


def test_explicit_remember_request_is_kept_after_long_history_and_restart(tmp_path):
    path = tmp_path / "history.sqlite3"
    memory = ConversationManager(storage_path=path)
    memory.add_user_message("Please remember that my preferred nickname is Nova.")
    for i in range(105):
        memory.add_user_message(f"Unrelated request {i}")
    restarted = ConversationManager(storage_path=path)
    restarted.new_conversation()
    restarted.add_user_message("Hello again")
    assert "preferred nickname is Nova" in str(restarted.get_context_messages())
