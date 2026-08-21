import logging
import sys

import httpx
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("mcp_weather")

mcp = FastMCP("Weather Service")


@mcp.tool()
async def get_weather(city_name: str) -> str:
    """
    Get the current weather forecast for a given city name.
    """
    try:
        async with httpx.AsyncClient() as client:
            # 1. Geocode the city name to get lat/lon
            geo_res = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city_name, "count": 1, "language": "en", "format": "json"},
                timeout=10.0,
            )
            geo_res.raise_for_status()
            geo_data = geo_res.json()

            results = geo_data.get("results")
            if not results:
                return f"Could not find coordinates for city: {city_name}"

            lat = results[0]["latitude"]
            lon = results[0]["longitude"]
            country = results[0].get("country", "")

            # 2. Get the weather data
            weather_res = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={"latitude": lat, "longitude": lon, "current_weather": "true"},
                timeout=10.0,
            )
            weather_res.raise_for_status()
            weather_data = weather_res.json()

            current = weather_data.get("current_weather", {})
            if not current:
                return f"Weather data not available for {city_name}."

            temp = current.get("temperature")
            windspeed = current.get("windspeed")
            weathercode = current.get("weathercode")

            # 3. Map WMO weather codes to human-readable strings
            conditions = {
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
            condition = conditions.get(weathercode, "Unknown conditions")

            return f"Current weather in {city_name} ({country}): {temp}°C, {condition}, Windspeed: {windspeed} km/h"

    except Exception as e:
        logger.error(f"Error fetching weather: {e}")
        return f"Failed to get weather data for {city_name}: {e}"


if __name__ == "__main__":
    logger.info("Starting Weather MCP Server...")
    mcp.run()
