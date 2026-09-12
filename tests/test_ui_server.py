"""Focused tests for dashboard data refresh behavior."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import websockets

from jarvis.config.settings import Settings
from jarvis.core.conversation import ConversationManager
from jarvis.tools.calendar_tool import GoogleCalendarTool
from jarvis.ui.server import WebSocketServer


def _server(weather_city: str = "") -> WebSocketServer:
    settings = SimpleNamespace(system=SimpleNamespace(weather_city=weather_city))
    return WebSocketServer(SimpleNamespace(settings=settings))


@pytest.mark.asyncio
async def test_weather_uses_location_fallback_and_current_api():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "ipwho.is":
            return httpx.Response(503, request=request)
        if request.url.host == "ipapi.co":
            return httpx.Response(
                200,
                request=request,
                json={
                    "latitude": 25.2048,
                    "longitude": 55.2708,
                    "city": "Dubai",
                    "country": "United Arab Emirates",
                },
            )
        if request.url.host == "api.open-meteo.com":
            return httpx.Response(
                200,
                request=request,
                json={
                    "current": {
                        "temperature_2m": 33.2,
                        "apparent_temperature": 38.0,
                        "weather_code": 1,
                        "wind_speed_10m": 12.5,
                    }
                },
            )
        raise AssertionError(f"Unexpected URL: {request.url}")

    server = _server()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await server._refresh_weather(client)

    weather = server.cached_world_data["weather"]
    assert "Dubai" in weather
    assert "33.2°C" in weather
    assert "Mainly clear" in weather


@pytest.mark.asyncio
async def test_calendar_dashboard_check_is_non_interactive(monkeypatch):
    interactive_values: list[bool] = []

    def authenticate(self, interactive: bool = True) -> bool:
        interactive_values.append(interactive)
        return False

    monkeypatch.setattr(GoogleCalendarTool, "authenticate", authenticate)
    server = _server()

    await server._refresh_schedule()

    assert interactive_values == [False]
    assert "not connected" in server.cached_world_data["schedule"]


@pytest.mark.asyncio
async def test_saved_history_roundtrip_clear_and_busy_guard(tmp_path):
    settings = Settings(gemini_api_key="", groq_api_key="")
    settings.logging.file = tmp_path / "jarvis.log"
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text('{"role":"user","text":"Saved telescope preference"}\n')
    history = ConversationManager(storage_path=tmp_path / "history.sqlite3", transcript=transcript)
    session = SimpleNamespace(
        settings=settings, router=SimpleNamespace(providers={}), memory=None, conversation=history
    )
    server = WebSocketServer(session, port=0)
    server._update_world_data_loop = AsyncMock()
    await server.start()
    try:
        async with websockets.connect(
            f"ws://127.0.0.1:{server.port}/?token={server.auth_token}"
        ) as ws:
            await ws.recv()
            await ws.send(json.dumps({"type": "get_history"}))
            result = json.loads(await ws.recv())
            assert result["messages"] == [{"role": "user", "text": "Saved telescope preference"}]
            async with server._chat_lock:
                await ws.send(json.dumps({"type": "clear_history"}))
                assert json.loads(await ws.recv())["type"] == "error"
                assert transcript.exists()
            await ws.send(json.dumps({"type": "clear_history"}))
            assert json.loads(await ws.recv())["type"] == "history_cleared"
            assert not transcript.exists()
            await ws.send(json.dumps({"type": "get_history"}))
            assert json.loads(await ws.recv())["messages"] == []
            restarted = ConversationManager(
                storage_path=tmp_path / "history.sqlite3", transcript=transcript
            )
            restarted.add_user_message("Remember telescope preference?")
            assert "Saved telescope preference" not in str(restarted.get_context_messages())
    finally:
        await server.stop()
