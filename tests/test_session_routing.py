"""Regression tests for the Groq-first session routing policy."""

from unittest.mock import AsyncMock, MagicMock

import pytest

import jarvis.core.session as session_module
from jarvis.config.settings import Settings
from jarvis.core.conversation import ConversationManager
from jarvis.core.session import SessionManager
from jarvis.llm.base import ModelTier


def test_session_intent_reserves_gemini_for_complex_or_native_tools():
    session = SessionManager()

    assert session._classify_intent_tier("Explain Python decorators") is ModelTier.STANDARD
    assert session._classify_intent_tier("Compare these two files") is ModelTier.STANDARD
    assert session._classify_intent_tier("Show my calendar") is ModelTier.STANDARD
    assert session._classify_intent_tier("Draft an email to Alex") is ModelTier.STANDARD
    assert session._classify_intent_tier("Analyze in detail this architecture") is ModelTier.COMPLEX
    assert session._classify_intent_tier("Open calculator") is ModelTier.COMPLEX


def test_live_context_query_carries_weather_across_confirmation():
    session = SessionManager()
    session.conversation = ConversationManager()
    session.conversation.add_user_message("Javis, what's the climate tomorrow?")
    session.conversation.add_assistant_message("I can fetch that forecast.")
    session.conversation.add_user_message("Yeah, you can do that.")

    resolved = session._resolve_context_query("Yeah, you can do that.")

    assert "climate tomorrow" in resolved
    assert "Yeah, you can do that" in resolved


@pytest.mark.parametrize(
    "phrase",
    [
        "Open Global Tab",
        "World Desk",
        "Open the global dashboard",
        "Open the file where I can see my CPU speed and all",
    ],
)
def test_world_monitor_aliases_are_deterministic(phrase):
    assert SessionManager._is_world_monitor_request(phrase) is True


@pytest.mark.asyncio
async def test_world_monitor_stream_emits_ui_action(monkeypatch):
    session = SessionManager()
    session._initialized = True
    session.router = MagicMock()
    session.conversation = ConversationManager()
    monkeypatch.setattr(session, "_append_transcript", MagicMock())

    chunks = [chunk async for chunk in session.process_input_stream("Open global dashboard")]

    assert {"__ui_action__": "open_world_monitor"} in chunks
    assert "Opened the World Monitor dashboard." in chunks


@pytest.mark.asyncio
async def test_initialization_maps_fast_and_standard_to_groq(monkeypatch, tmp_path):
    settings = Settings(gemini_api_key="gemini-test", groq_api_key="groq-test")
    settings.memory.data_dir = tmp_path / "data"
    settings.logging.file = tmp_path / "logs" / "jarvis.log"
    gemini = MagicMock()
    gemini.name = "gemini"
    gemini.health_check = AsyncMock(return_value=True)
    groq = MagicMock()
    groq.name = "groq"
    groq.health_check = AsyncMock(return_value=True)

    monkeypatch.setattr(session_module, "get_settings", lambda: settings)
    monkeypatch.setattr(session_module, "setup_logging", lambda _settings: None)
    gemini_factory = MagicMock(return_value=gemini)
    monkeypatch.setattr(session_module, "GeminiProvider", gemini_factory)
    monkeypatch.setattr(session_module, "GroqProvider", MagicMock(return_value=groq))
    mcp = MagicMock()
    mcp.start_server = AsyncMock(return_value=False)
    monkeypatch.setattr(session_module, "MCPManager", MagicMock(return_value=mcp))

    session = SessionManager()
    await session.initialize()

    assert session.router is not None
    assert session.router.providers[ModelTier.FAST] is groq
    assert session.router.providers[ModelTier.STANDARD] is groq
    assert session.router.providers[ModelTier.COMPLEX] is gemini
    assert gemini_factory.call_count == 1
