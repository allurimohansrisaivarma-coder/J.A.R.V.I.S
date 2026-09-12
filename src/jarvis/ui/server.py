import asyncio
import html
import json
import re
import secrets
import time
from contextlib import suppress
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import psutil
import structlog
import websockets
from websockets.exceptions import ConnectionClosed

from jarvis.utils.errors import user_error
from jarvis.utils.metrics import metrics

logger = structlog.get_logger(__name__)


class WebSocketServer:
    def __init__(self, session_manager, host="127.0.0.1", port=8741, voice_manager=None):
        self.session_manager = session_manager
        self.host = host
        self.port = port
        self.auth_token = secrets.token_urlsafe(32)
        self.clients: set[Any] = set()
        self._server = None
        self.voice_manager = voice_manager
        self.hotkey_listener = None
        self.hotkey_queue = None
        self.server_loop = None
        self._loop = None  # captured in start()
        self._chat_lock = asyncio.Lock()
        self._chat_tasks: set[asyncio.Task[Any]] = set()
        self._world_task: asyncio.Task[None] | None = None
        self._world_refresh_lock = asyncio.Lock()
        self._last_world_refresh = 0.0
        self.cached_world_data = {
            "weather": "Resolving local weather...",
            "news": "Loading...",
            "schedule": "Checking calendar connection...",
        }

    async def start(self):
        """Start the WebSocket server and wire up voice callbacks."""
        self._loop = asyncio.get_running_loop()
        logger.info("Starting WebSocket server", host=self.host, port=self.port)

        # Wire voice manager callbacks now that we have a guaranteed loop reference
        if self.voice_manager:
            loop = self._loop

            self.voice_manager.on_state_change = lambda s: asyncio.run_coroutine_threadsafe(
                self.broadcast_state(s.name), loop
            )
            self.voice_manager.on_user_message = lambda t: asyncio.run_coroutine_threadsafe(
                self.broadcast(json.dumps({"type": "user_msg", "text": t})), loop
            )

            def handle_chunk(c):
                if isinstance(c, dict):
                    if "__terminal__" in c:
                        asyncio.run_coroutine_threadsafe(
                            self.broadcast(
                                json.dumps({"type": "terminal", "text": c["__terminal__"]})
                            ),
                            loop,
                        )
                    elif "__ui_action__" in c:
                        asyncio.run_coroutine_threadsafe(
                            self.broadcast(json.dumps({"type": c["__ui_action__"]})), loop
                        )
                else:
                    asyncio.run_coroutine_threadsafe(
                        self.broadcast(json.dumps({"type": "chunk", "text": c})), loop
                    )

            self.voice_manager.on_jarvis_chunk = handle_chunk
            self.voice_manager.on_jarvis_done = lambda: asyncio.run_coroutine_threadsafe(
                self.broadcast(json.dumps({"type": "done"})), loop
            )
            self.voice_manager.on_error = lambda message: asyncio.run_coroutine_threadsafe(
                self.broadcast(json.dumps({"type": "error", "message": message})), loop
            )
            self.voice_manager.tts.on_unavailable = lambda message: (
                asyncio.run_coroutine_threadsafe(
                    self.broadcast(json.dumps({"type": "notice", "message": message})), loop
                )
            )

        self.session_manager.on_metadata = lambda meta: asyncio.run_coroutine_threadsafe(
            self.broadcast(json.dumps({"type": "message_meta", "meta": meta})), self._loop
        )

        try:
            self._server = await websockets.serve(
                self.handle_client, self.host, self.port, max_size=65536
            )
        except OSError as exc:
            if exc.errno not in (10048, 98) and getattr(exc, "winerror", None) != 10048:
                raise
            logger.warning("Configured port occupied; selecting an available port")
            self._server = await websockets.serve(self.handle_client, self.host, 0, max_size=65536)
        self.port = self._server.sockets[0].getsockname()[1]
        self._world_task = asyncio.create_task(self._update_world_data_loop())

    async def stop(self):
        if self.hotkey_listener:
            self.hotkey_listener.stop()
        if self.voice_manager:
            self.voice_manager.stop()
        self.session_manager.on_metadata = None

        tasks = [*self._chat_tasks]
        if self._world_task:
            tasks.append(self._world_task)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        if self._server:
            logger.info("Stopping WebSocket server")
            self._server.close()
            await self._server.wait_closed()

    async def broadcast(self, message: str):
        async def send(client: Any) -> None:
            try:
                await client.send(message)
            except ConnectionClosed:
                self.clients.discard(client)

        await asyncio.gather(*(send(client) for client in list(self.clients)))

    async def broadcast_state(self, state: str):
        await self.broadcast(json.dumps({"type": "state", "state": state}))

    @staticmethod
    def _weather_condition(code: int | None) -> str:
        if code is None:
            return "Current conditions"
        conditions = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Rime fog",
            51: "Light drizzle",
            53: "Drizzle",
            55: "Dense drizzle",
            61: "Light rain",
            63: "Rain",
            65: "Heavy rain",
            71: "Light snow",
            73: "Snow",
            75: "Heavy snow",
            80: "Rain showers",
            81: "Rain showers",
            82: "Heavy showers",
            95: "Thunderstorm",
            96: "Thunderstorm with hail",
            99: "Severe thunderstorm with hail",
        }
        return conditions.get(code, "Current conditions")

    async def _resolve_weather_location(
        self, client: httpx.AsyncClient
    ) -> tuple[float, float, str] | None:
        configured_city = self.session_manager.settings.system.weather_city.strip()
        if configured_city:
            response = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": configured_city, "count": 1, "language": "en", "format": "json"},
                timeout=5.0,
            )
            response.raise_for_status()
            results = response.json().get("results") or []
            if results:
                location = results[0]
                label = ", ".join(
                    part for part in (location.get("name"), location.get("country")) if part
                )
                return float(location["latitude"]), float(location["longitude"]), label
            raise ValueError(f"Weather city was not found: {configured_city}")

        providers = (
            ("https://ipwho.is/", "latitude", "longitude", "success"),
            ("https://ipapi.co/json/", "latitude", "longitude", None),
        )
        for url, lat_key, lon_key, success_key in providers:
            try:
                response = await client.get(
                    url, headers={"User-Agent": "JARVIS/0.1 local-weather"}, timeout=4.0
                )
                response.raise_for_status()
                location = response.json()
                if success_key and location.get(success_key) is False:
                    continue
                latitude = location.get(lat_key)
                longitude = location.get(lon_key)
                if latitude is None or longitude is None:
                    continue
                label = ", ".join(
                    str(part) for part in (location.get("city"), location.get("country")) if part
                )
                return float(latitude), float(longitude), label or "Local area"
            except Exception as exc:
                logger.debug("Location provider failed", provider=url, error=str(exc))
        return None

    async def _refresh_weather(self, client: httpx.AsyncClient) -> None:
        try:
            location = await self._resolve_weather_location(client)
            if location is None:
                raise RuntimeError("automatic location detection failed")
            latitude, longitude, label = location
            response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": ("temperature_2m,apparent_temperature,weather_code,wind_speed_10m"),
                    "timezone": "auto",
                },
                timeout=6.0,
            )
            response.raise_for_status()
            current = response.json().get("current") or {}
            temperature = current.get("temperature_2m")
            if temperature is None:
                raise RuntimeError("weather response did not contain a temperature")
            condition = self._weather_condition(current.get("weather_code"))
            feels_like = current.get("apparent_temperature")
            wind = current.get("wind_speed_10m")
            details = [f"{html.escape(str(temperature))}°C", html.escape(condition)]
            if feels_like is not None:
                details.append(f"Feels {html.escape(str(feels_like))}°C")
            if wind is not None:
                details.append(f"Wind {html.escape(str(wind))} km/h")
            self.cached_world_data["weather"] = f"<b>{html.escape(label)}</b><br>" + " · ".join(
                details
            )
        except Exception as exc:
            logger.warning("Weather update failed", error=str(exc))
            self.cached_world_data["weather"] = (
                "Weather unavailable. Set a city in Settings to bypass automatic location detection."
            )

    async def _refresh_news(self, client: httpx.AsyncClient) -> None:
        try:
            response = await client.get(
                "https://hacker-news.firebaseio.com/v0/topstories.json", timeout=5.0
            )
            response.raise_for_status()
            story_ids = response.json()[:3]

            async def fetch_story(story_id: int) -> str:
                story = await client.get(
                    f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json",
                    timeout=5.0,
                )
                story.raise_for_status()
                return html.escape(str((story.json() or {}).get("title", "Untitled")))

            titles = await asyncio.gather(*(fetch_story(story_id) for story_id in story_ids))
            items = "".join(f"<li>{title}</li>" for title in titles)
            self.cached_world_data["news"] = f"<ul class='wm-news-list'>{items}</ul>"
        except Exception as exc:
            logger.warning("News update failed", error=str(exc))
            self.cached_world_data["news"] = "Global intelligence feed is temporarily unavailable."

    async def _refresh_schedule(self) -> None:
        try:
            from jarvis.tools.calendar_tool import GoogleCalendarTool

            calendar = GoogleCalendarTool()
            authenticated = await asyncio.to_thread(calendar.authenticate, False)
            if not authenticated:
                self.cached_world_data["schedule"] = (
                    "Google Calendar is not connected. Ask JARVIS about your calendar to start OAuth."
                )
                return
            events = await asyncio.to_thread(calendar.get_upcoming_events, 5)
            self.cached_world_data["schedule"] = html.escape(events).replace("\n", "<br>")
        except Exception as exc:
            logger.warning("Calendar dashboard update failed", error=str(exc))
            self.cached_world_data["schedule"] = "Google Calendar could not be refreshed."

    async def _refresh_world_data(self, *, force: bool = False) -> None:
        requested_at = time.monotonic()
        async with self._world_refresh_lock:
            # A refresh that completed while this caller waited already
            # satisfies the request and avoids a second round of API calls.
            if self._last_world_refresh >= requested_at:
                return
            if not force and time.monotonic() - self._last_world_refresh < 60:
                return
            async with httpx.AsyncClient(follow_redirects=True) as client:
                await asyncio.gather(
                    self._refresh_weather(client),
                    self._refresh_news(client),
                    self._refresh_schedule(),
                )
            self._last_world_refresh = time.monotonic()

    async def _update_world_data_loop(self):
        """Background task to refresh dashboard data periodically."""
        while True:
            try:
                await self._refresh_world_data(force=True)
            except Exception as exc:
                logger.error("Failed to update world data", error=str(exc))
            await asyncio.sleep(900)  # update every 15 mins

    async def handle_client(self, websocket):
        token = parse_qs(urlparse(websocket.request.path).query).get("token", [""])[0]
        if not secrets.compare_digest(token, self.auth_token):
            await websocket.close(code=1008, reason="Unauthorized")
            return
        self.clients.add(websocket)
        logger.info("Client connected", clients=len(self.clients))

        try:
            await websocket.send(
                json.dumps(
                    {
                        "type": "capabilities",
                        "chat": bool(self.session_manager.router.providers),
                        "voice": self.voice_manager is not None,
                    }
                )
            )
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if not isinstance(data, dict):
                        raise TypeError("Request must be an object")
                    msg_type = data.get("type")

                    if msg_type == "chat":
                        text = data.get("text", "")
                        if isinstance(text, str) and text.strip():
                            task = asyncio.create_task(self.process_chat(websocket, text.strip()))
                            self._chat_tasks.add(task)
                            task.add_done_callback(self._chat_tasks.discard)
                    elif msg_type == "voice_start":
                        logger.info("Voice start requested (PTT)")
                        if self.voice_manager:
                            # A held mic press has priority over JARVIS audio.
                            # It stops playback before capture begins, then the
                            # PTT loop records until the button is released.
                            self.voice_manager.interrupt_speech()
                            self.voice_manager.on_audio_level = lambda vol: asyncio.create_task(
                                self.broadcast(json.dumps({"type": "audio_level", "level": vol}))
                            )
                            # Start the PTT loop if not already running
                            if not self.voice_manager._is_running:
                                asyncio.create_task(self.voice_manager.start_ptt_loop())
                                # Give the loop a moment to start waiting on the queue
                                await asyncio.sleep(0.05)
                            # Signal recording to begin
                            await self.voice_manager.hotkey_queue.put("PTT_START")
                        else:
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": "error",
                                        "message": "Voice input needs a Groq key and an available microphone. Open Settings to configure it.",
                                    }
                                )
                            )
                            await self.broadcast_state("IDLE")
                    elif msg_type == "voice_stop":
                        logger.info("Voice stop requested (PTT)")
                        if self.voice_manager:
                            await self.voice_manager.hotkey_queue.put("PTT_STOP")
                    elif msg_type == "cancel":
                        for task in list(self._chat_tasks):
                            task.cancel()
                        if self.voice_manager:
                            await self.voice_manager.cancel_turn()
                        await self.broadcast(json.dumps({"type": "done"}))
                        await self.broadcast_state("IDLE")
                    elif msg_type == "get_history":
                        conversation = getattr(self.session_manager, "conversation", None)
                        active = conversation.get_active() if conversation else None
                        history = (
                            [
                                {"role": m.role, "text": m.content}
                                for m in active.messages[-40:]
                                if m.role in {"user", "assistant"} and isinstance(m.content, str)
                            ]
                            if active
                            else []
                        )
                        await websocket.send(
                            json.dumps({"type": "history_data", "messages": history})
                        )
                    elif msg_type == "clear_history":
                        if self._chat_lock.locked() or (
                            self.voice_manager
                            and (
                                self.voice_manager.tts.is_speaking
                                or self.voice_manager.state.name != "IDLE"
                            )
                        ):
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": "error",
                                        "message": "Stop the active reply before clearing conversation history.",
                                    }
                                )
                            )
                            continue
                        conversation = getattr(self.session_manager, "conversation", None)
                        if conversation:
                            conversation.clear_history()
                            for filename in ("transcript.txt", "transcript.jsonl"):
                                (
                                    self.session_manager.settings.logging.file.parent / filename
                                ).unlink(missing_ok=True)
                        await websocket.send(json.dumps({"type": "history_cleared"}))
                    elif msg_type == "get_settings":
                        cfg = self.session_manager.settings.voice
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "settings_data",
                                    "settings": {
                                        "push_to_talk_key": cfg.push_to_talk_key,
                                        "silence_duration": cfg.silence_duration,
                                        "tts_voice": cfg.tts_voice,
                                        "tts_rate": cfg.tts_rate,
                                        "weather_city": (
                                            self.session_manager.settings.system.weather_city
                                        ),
                                    },
                                }
                            )
                        )
                    elif msg_type == "update_settings":
                        cfg = self.session_manager.settings.voice
                        new_settings = data.get("settings", {})
                        rate_input = str(new_settings.get("tts_rate", cfg.tts_rate)).strip()
                        if not re.fullmatch(r"[+-]?\d{1,3}%?", rate_input):
                            raise ValueError("Use a speech rate such as +20%")

                        cfg.push_to_talk_key = str(
                            new_settings.get("push_to_talk_key", cfg.push_to_talk_key)
                        )
                        silence_duration = new_settings.get(
                            "silence_duration", cfg.silence_duration
                        )
                        cfg.silence_duration = min(5.0, max(0.2, float(silence_duration)))
                        cfg.tts_voice = str(new_settings.get("tts_voice", cfg.tts_voice))

                        rate = str(new_settings.get("tts_rate", cfg.tts_rate)).strip()
                        if not rate.endswith("%"):
                            rate += "%"
                        if not rate.startswith("+") and not rate.startswith("-"):
                            rate = f"+{rate}"
                        cfg.tts_rate = rate
                        self.session_manager.settings.system.weather_city = str(
                            new_settings.get(
                                "weather_city",
                                self.session_manager.settings.system.weather_city,
                            )
                        ).strip()

                        if self.voice_manager:
                            self.voice_manager.tts.voice = cfg.tts_voice
                            self.voice_manager.tts.rate = cfg.tts_rate

                        self.session_manager.settings.save_to_yaml(self.session_manager.config_path)
                        task = asyncio.create_task(self._refresh_world_data(force=True))
                        self._chat_tasks.add(task)
                        task.add_done_callback(self._chat_tasks.discard)
                        await websocket.send(json.dumps({"type": "settings_saved"}))

                        hotkey_listener = self.hotkey_listener
                        server_loop = self.server_loop
                        hotkey_queue = self.hotkey_queue
                        if hotkey_listener and server_loop and hotkey_queue:
                            from jarvis.voice.hotkey import HotkeyListener

                            if hotkey_listener.hotkey_combo != cfg.push_to_talk_key.lower():
                                hotkey_listener.stop()
                                self.hotkey_listener = HotkeyListener(
                                    hotkey_combo=cfg.push_to_talk_key
                                )
                                self.hotkey_listener.start(server_loop, hotkey_queue)
                                logger.info("Dynamic hotkey reload complete")
                    elif msg_type == "get_stats":
                        uptime = 0.0
                        if hasattr(metrics, "start_time"):
                            uptime = time.time() - metrics.start_time

                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "stats_data",
                                    "stats": {
                                        "uptime": uptime,
                                        "total_tokens": metrics.total_tokens,
                                    },
                                }
                            )
                        )
                    elif msg_type == "get_world_monitor":
                        if data.get("refresh"):
                            task = asyncio.create_task(self._refresh_world_data(force=True))
                            self._chat_tasks.add(task)
                            task.add_done_callback(self._chat_tasks.discard)
                        ram = psutil.virtual_memory().percent
                        cpu = psutil.cpu_percent()

                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "world_monitor_data",
                                    "data": {
                                        "ram": ram,
                                        "cpu": cpu,
                                        "weather": self.cached_world_data.get("weather", "N/A"),
                                        "news": self.cached_world_data.get("news", "N/A"),
                                        "schedule": self.cached_world_data.get("schedule", "N/A"),
                                    },
                                }
                            )
                        )
                    elif msg_type == "get_memories":
                        if not self.session_manager.memory:
                            await websocket.send(
                                json.dumps(
                                    {"type": "memories_data", "memories": [], "disabled": True}
                                )
                            )
                        if self.session_manager.memory:
                            time_filter = data.get("filter", "all")
                            from jarvis.memory.database import get_sqlite_session
                            from jarvis.memory.schema import Fact

                            try:
                                db = get_sqlite_session()
                                facts = db.query(Fact).all()
                                memories = []
                                now = time.time()
                                max_age = 86400 if time_filter == "24h" else 3600

                                for f in facts:
                                    # SQLite fact created_at is a datetime object
                                    created_ts = f.created_at.timestamp() if f.created_at else 0
                                    if time_filter == "all" or now - created_ts <= max_age:
                                        memories.append({"id": f.id, "fact": f.content})

                                await websocket.send(
                                    json.dumps({"type": "memories_data", "memories": memories})
                                )
                            finally:
                                if "db" in locals():
                                    db.close()
                    elif msg_type == "delete_memory":
                        if self.session_manager.memory:
                            mem_id = data.get("id")
                            if mem_id:
                                try:
                                    from jarvis.memory.database import get_sqlite_session
                                    from jarvis.memory.schema import Fact

                                    db = get_sqlite_session()
                                    try:
                                        fact = db.query(Fact).filter(Fact.id == mem_id).first()
                                        if fact:
                                            target_content = fact.content
                                            db.delete(fact)
                                            db.commit()

                                            safe_content = target_content.replace("'", "''")
                                            try:
                                                self.session_manager.memory.table.delete(
                                                    f"content = '{safe_content}'"
                                                )
                                            except Exception as e:
                                                logger.warning(
                                                    "Failed to delete from vector db", error=str(e)
                                                )

                                            await websocket.send(
                                                json.dumps({"type": "memory_deleted", "id": mem_id})
                                            )
                                    finally:
                                        db.close()
                                except Exception as e:
                                    logger.error("Failed to delete memory", error=str(e))
                    elif msg_type == "add_memory":
                        if not self.session_manager.memory:
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": "error",
                                        "message": "Long-term memory is disabled in configuration.",
                                    }
                                )
                            )
                        if self.session_manager.memory:
                            fact = data.get("fact")
                            if fact:
                                try:
                                    await asyncio.to_thread(
                                        self.session_manager.memory.store_fact, fact
                                    )
                                    await websocket.send(json.dumps({"type": "memory_added"}))
                                except Exception as e:
                                    logger.error("Failed to add memory", error=str(e))
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                    await websocket.send(
                        json.dumps({"type": "error", "message": "Invalid request"})
                    )
                except Exception as e:
                    logger.error("Error processing message", error=str(e))
                    await websocket.send(
                        json.dumps({"type": "error", "message": "Request could not be processed"})
                    )
        except ConnectionClosed:
            logger.info("Client disconnected")
        finally:
            self.clients.discard(websocket)
            if not self.clients and self.voice_manager:
                await self.voice_manager.hotkey_queue.put("PTT_STOP")

    async def process_chat(self, websocket, text: str):
        """Handle one turn at a time so chat and audio cannot interleave."""
        if self._chat_lock.locked():
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "message": "A reply is in progress. Press Stop before sending another request.",
                    }
                )
            )
            return
        async with self._chat_lock:
            try:
                if self.voice_manager:
                    await self.voice_manager.respond_to_text(text)
                else:
                    await self.broadcast_state("THINKING")
                    async for chunk in self.session_manager.process_input_stream(text):
                        if isinstance(chunk, dict) and "__terminal__" in chunk:
                            await websocket.send(
                                json.dumps({"type": "terminal", "text": chunk["__terminal__"]})
                            )
                        elif isinstance(chunk, dict) and "__ui_action__" in chunk:
                            await websocket.send(json.dumps({"type": chunk["__ui_action__"]}))
                        else:
                            if isinstance(chunk, str):
                                await websocket.send(json.dumps({"type": "chunk", "text": chunk}))
                    await websocket.send(json.dumps({"type": "done"}))
            except asyncio.CancelledError:
                logger.info("Chat response interrupted")
                with suppress(ConnectionClosed):
                    await websocket.send(json.dumps({"type": "done"}))
                raise
            except Exception as e:
                logger.error("Error processing chat", error=str(e))
                with suppress(ConnectionClosed):
                    await websocket.send(json.dumps({"type": "error", "message": user_error(e)}))
                    await websocket.send(json.dumps({"type": "done"}))
                await self.broadcast_state("ERROR")
            finally:
                await self.broadcast_state("IDLE")
