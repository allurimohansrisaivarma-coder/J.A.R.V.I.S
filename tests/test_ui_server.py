"""Focused tests for dashboard data refresh behavior."""

from types import SimpleNamespace

import httpx
import pytest

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
