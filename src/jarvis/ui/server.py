import asyncio
import json
import structlog
import websockets
from websockets.exceptions import ConnectionClosed

logger = structlog.get_logger(__name__)

class WebSocketServer:
    def __init__(self, session_manager, host="127.0.0.1", port=8741, voice_manager=None):
        self.session_manager = session_manager
        self.host = host
        self.port = port
        self.clients = set()
        self._server = None
        self.voice_manager = voice_manager
        self.hotkey_listener = None
        self.hotkey_queue = None
        self.server_loop = None
        self._loop = None  # captured in start()
        self._chat_lock = asyncio.Lock()

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
            self.voice_manager.on_jarvis_chunk = lambda c: asyncio.run_coroutine_threadsafe(
                self.broadcast(json.dumps({"type": "chunk", "text": c})), loop
            )
            self.voice_manager.on_jarvis_done = lambda: asyncio.run_coroutine_threadsafe(
                self.broadcast(json.dumps({"type": "done"})), loop
            )
        
        self._server = await websockets.serve(self.handle_client, self.host, self.port)

    async def stop(self):
        if self._server:
            logger.info("Stopping WebSocket server")
            self._server.close()
            await self._server.wait_closed()

    async def broadcast(self, message: str):
        for client in list(self.clients):
            try:
                await client.send(message)
            except ConnectionClosed:
                pass

    async def broadcast_state(self, state: str):
        await self.broadcast(json.dumps({"type": "state", "state": state}))

    async def handle_client(self, websocket):
        self.clients.add(websocket)
        logger.info("Client connected")
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    
                    if msg_type == "chat":
                        text = data.get("text", "")
                        if text:
                            asyncio.create_task(self.process_chat(websocket, text))
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
                    elif msg_type == "voice_stop":
                        logger.info("Voice stop requested (PTT)")
                        if self.voice_manager:
                            await self.voice_manager.hotkey_queue.put("PTT_STOP")
                    elif msg_type == "get_settings":
                        cfg = self.session_manager.settings.voice
                        await websocket.send(json.dumps({
                            "type": "settings_data",
                            "settings": {
                                "push_to_talk_key": cfg.push_to_talk_key,
                                "silence_duration": cfg.silence_duration,
                                "tts_voice": cfg.tts_voice,
                                "tts_rate": cfg.tts_rate,
                            }
                        }))
                    elif msg_type == "update_settings":
                        cfg = self.session_manager.settings.voice
                        new_settings = data.get("settings", {})
                        
                        cfg.push_to_talk_key = str(new_settings.get("push_to_talk_key", cfg.push_to_talk_key))
                        silence_duration = new_settings.get("silence_duration", cfg.silence_duration)
                        cfg.silence_duration = min(5.0, max(0.2, float(silence_duration)))
                        cfg.tts_voice = str(new_settings.get("tts_voice", cfg.tts_voice))
                        cfg.tts_rate = str(new_settings.get("tts_rate", cfg.tts_rate))

                        if self.voice_manager:
                            self.voice_manager.tts.voice = cfg.tts_voice
                            self.voice_manager.tts.rate = cfg.tts_rate
                        
                        self.session_manager.settings.save_to_yaml()
                        
                        if getattr(self, 'hotkey_listener', None) and getattr(self, 'server_loop', None):
                            from jarvis.voice.hotkey import HotkeyListener
                            if self.hotkey_listener.hotkey_combo != cfg.push_to_talk_key.lower():
                                self.hotkey_listener.stop()
                                self.hotkey_listener = HotkeyListener(hotkey_combo=cfg.push_to_talk_key)
                                self.hotkey_listener.start(self.server_loop, self.hotkey_queue)
                                logger.info("Dynamic hotkey reload complete")
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                except Exception as e:
                    logger.error("Error processing message", error=str(e))
        except ConnectionClosed:
            logger.info("Client disconnected")
        finally:
            self.clients.discard(websocket)

    async def process_chat(self, websocket, text: str):
        """Handle one turn at a time so chat and audio cannot interleave."""
        async with self._chat_lock:
            try:
                if self.voice_manager:
                    await self.voice_manager.respond_to_text(text)
                else:
                    await self.broadcast_state("THINKING")
                    async for chunk in self.session_manager.process_input_stream(text):
                        await websocket.send(json.dumps({"type": "chunk", "text": chunk}))
                    await websocket.send(json.dumps({"type": "done"}))
                await self.broadcast_state("IDLE")
            except asyncio.CancelledError:
                logger.info("Chat response interrupted")
                await websocket.send(json.dumps({"type": "done"}))
                await self.broadcast_state("IDLE")
            except Exception as e:
                logger.error("Error processing chat", error=str(e))
                await websocket.send(json.dumps({"type": "chunk", "text": "\n[Error: Could not process request]"}))
                await websocket.send(json.dumps({"type": "done"}))
                await self.broadcast_state("ERROR")
