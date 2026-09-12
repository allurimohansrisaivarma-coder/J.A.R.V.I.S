import logging
import sys

import httpx
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("mcp_weather")

mcp = FastMCP("Weather Service")

_CONDITIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


async def _resolve_location(
    client: httpx.AsyncClient, city_name: str
) -> tuple[float, float, str] | None:
    if city_name:
        response = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city_name, "count": 1, "language": "en", "format": "json"},
            timeout=6.0,
        )
        response.raise_for_status()
        results = response.json().get("results") or []
        if not results:
            return None
        location = results[0]
        label = ", ".join(
            str(part) for part in (location.get("name"), location.get("country")) if part
        )
        return float(location["latitude"]), float(location["longitude"]), label

    for url, success_key in (("https://ipwho.is/", "success"), ("https://ipapi.co/json/", None)):
        try:
            response = await client.get(url, headers={"User-Agent": "JARVIS/0.2"}, timeout=4.0)
            response.raise_for_status()
            location = response.json()
            if success_key and location.get(success_key) is False:
                continue
            latitude = location.get("latitude")
            longitude = location.get("longitude")
            if latitude is None or longitude is None:
                continue
            label = ", ".join(
                str(part) for part in (location.get("city"), location.get("country")) if part
            )
            return float(latitude), float(longitude), label or "your area"
        except Exception as exc:
            logger.debug("Location provider failed for %s: %s", url, exc)
            continue
    return None


@mcp.tool()
async def get_weather(city_name: str = "", day: str = "current") -> str:
    """
    Get current or tomorrow's live forecast. An empty city uses approximate IP location.
    """
    try:
        async with httpx.AsyncClient() as client:
            location = await _resolve_location(client, city_name.strip())
            if location is None:
                return f"Could not determine a weather location for {city_name or 'your area'}."
            latitude, longitude, label = location

            weather_res = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                    "daily": (
                        "weather_code,temperature_2m_max,temperature_2m_min,"
                        "precipitation_probability_max,wind_speed_10m_max"
                    ),
                    "forecast_days": 7,
                    "timezone": "auto",
                },
                timeout=10.0,
            )
            weather_res.raise_for_status()
            weather_data = weather_res.json()

            normalized_day = day.strip().casefold()
            if normalized_day == "tomorrow":
                daily = weather_data.get("daily") or {}
                dates = daily.get("time") or []
                if len(dates) < 2:
                    return f"Tomorrow's forecast is not available for {label}."
                index = 1
                code = (daily.get("weather_code") or [None, None])[index]
                high = (daily.get("temperature_2m_max") or [None, None])[index]
                low = (daily.get("temperature_2m_min") or [None, None])[index]
                rain = (daily.get("precipitation_probability_max") or [None, None])[index]
                wind = (daily.get("wind_speed_10m_max") or [None, None])[index]
                condition = _CONDITIONS.get(
                    int(code) if code is not None else -1, "Unknown conditions"
                )
                return (
                    f"Tomorrow's forecast for {label} ({dates[index]}): {condition}; "
                    f"low {low}°C, high {high}°C, precipitation chance {rain}%, "
                    f"maximum wind {wind} km/h."
                )

            current = weather_data.get("current") or {}
            temperature = current.get("temperature_2m")
            if temperature is None:
                return f"Current weather data is not available for {label}."
            current_code = current.get("weather_code")
            condition = _CONDITIONS.get(
                int(current_code) if current_code is not None else -1,
                "Unknown conditions",
            )
            return (
                f"Current weather in {label}: {temperature}°C, {condition}, "
                f"feels like {current.get('apparent_temperature')}°C, "
                f"wind {current.get('wind_speed_10m')} km/h."
            )

    except Exception as e:
        logger.error(f"Error fetching weather: {e}")
        return f"Failed to get weather data for {city_name}: {e}"


if __name__ == "__main__":
    logger.info("Starting Weather MCP Server...")
    mcp.run()
