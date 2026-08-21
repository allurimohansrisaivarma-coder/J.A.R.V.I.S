"""JARVIS Desktop UI — pywebview window with WebSocket bridge."""

import asyncio
import threading
from pathlib import Path

import structlog

from jarvis.ui.server import WebSocketServer

logger = structlog.get_logger(__name__)


def _build_index_url(index_path: Path, port: int) -> str:
    """Build a WebView-safe file URL while passing runtime data in the fragment."""
    resolved = index_path.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"JARVIS UI entry point is not a file: {resolved}")
    # Edge/WebView2 may interpret a query string on file:// URLs as part of
    # the filename. URL fragments are never sent to the filesystem.
    return f"{resolved.as_uri()}#port={port}"


class JarvisAPI:
    """Python ↔ JavaScript bridge exposed to the webview window."""

    def __init__(self, session_manager, server_loop=None, server=None):
        self._session_manager = session_manager
        self._window = None
        self._loop = server_loop
        self._server = server
        self._normal_size = (1200, 800)
        self._normal_on_top = False
        self._window_lock = threading.RLock()

    def minimize_to_tray(self):
        """Minimize the window."""
        if self._window:
            self._window.minimize()

    def collapse_to_widget(self) -> bool:
        """Collapse the window into a small floating widget."""
        if not self._window:
            return False
        with self._window_lock:
            try:
                self._normal_size = (
                    max(640, int(self._window.width)),
                    max(480, int(self._window.height)),
                )
                self._normal_on_top = bool(self._window.on_top)
                # Stay above pywebview's native minimum size. Resizing below
                # that boundary can terminate Edge/WebView2 on Windows.
                self._window.resize(112, 112)
                self._window.on_top = True
                return True
            except Exception as exc:
                logger.error("Failed to enter widget mode", error=str(exc))
                try:
                    self._window.resize(*self._normal_size)
                    self._window.on_top = self._normal_on_top
                except Exception:
                    logger.warning("Failed to restore window after widget-mode error")
                return False

    def expand_to_window(self) -> bool:
        """Expand the window to full UI size."""
        if not self._window:
            return False
        with self._window_lock:
            try:
                self._window.resize(*self._normal_size)
                self._window.on_top = self._normal_on_top
                return True
            except Exception as exc:
                logger.error("Failed to leave widget mode", error=str(exc))
                return False

    def toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        if self._window:
            self._window.toggle_fullscreen()

    def close_app(self):
        """Close the application cleanly."""
        if self._window:
            self._window.destroy()


def _run_server(server, loop, ready_event, startup_state):
    """Own the complete async application lifecycle on one background loop."""
    asyncio.set_event_loop(loop)

    async def lifecycle() -> None:
        # MCP's AnyIO streams must be opened and closed by the same task. Keep
        # the whole backend lifetime inside this one coroutine.
        shutdown_event = asyncio.Event()
        startup_state["shutdown_event"] = shutdown_event
        try:
            await server.session_manager.initialize()
            server.port = server.session_manager.settings.system.port

            if server.voice_manager is None and server.session_manager.settings.voice.enabled:
                api_key = server.session_manager.settings.primary_groq_key
                if api_key:
                    from jarvis.voice import VoiceManager
                    from jarvis.voice.stt import STTProvider
                    from jarvis.voice.tts import TTSProvider

                    voice_cfg = server.session_manager.settings.voice
                    server.voice_manager = VoiceManager(
                        server.session_manager,
                        STTProvider(
                            api_key=api_key,
                            model=voice_cfg.stt_model,
                            language=voice_cfg.stt_language,
                        ),
                        TTSProvider(
                            voice=voice_cfg.tts_voice,
                            rate=voice_cfg.tts_rate,
                            pitch=voice_cfg.tts_pitch,
                        ),
                    )
                else:
                    logger.warning("Voice disabled because no Groq API key is configured")

            await server.start()
            ready_event.set()
            await shutdown_event.wait()
        except Exception as exc:
            startup_state["error"] = exc
            ready_event.set()
        finally:
            await server.stop()
            await server.session_manager.shutdown()

    try:
        loop.run_until_complete(lifecycle())
    finally:
        loop.close()


def launch_ui(session_manager, voice_manager=None):
    """Launch the JARVIS desktop UI.

    This creates a WebSocket server on a background thread
    and opens the pywebview window on the main thread.

    Args:
        session_manager: Initialized SessionManager instance.
        voice_manager: Optional initialized VoiceManager instance for Live Conversation.
    """
    import webview

    logger.info("Launching JARVIS Desktop UI")

    # Start WebSocket server on a background thread with its own event loop
    server_loop = asyncio.new_event_loop()
    server = WebSocketServer(session_manager, voice_manager=voice_manager)
    ready_event = threading.Event()
    startup_state: dict[str, object] = {}
    server_thread = threading.Thread(
        target=_run_server, args=(server, server_loop, ready_event, startup_state), daemon=True
    )
    server_thread.start()

    if not ready_event.wait(timeout=60):
        server_loop.call_soon_threadsafe(server_loop.stop)
        server_thread.join(timeout=5)
        raise RuntimeError("Timed out while initializing the JARVIS backend")
    if "error" in startup_state:
        startup_error = startup_state["error"]
        if isinstance(startup_error, BaseException):
            raise RuntimeError("JARVIS backend initialization failed") from startup_error
        raise RuntimeError("JARVIS backend initialization failed")

    voice_manager = server.voice_manager

    # Create the API bridge
    api = JarvisAPI(session_manager)
    api._loop = server_loop
    api._server = server

    # Resolve the static files path
    import sys

    if hasattr(sys, "_MEIPASS"):
        static_dir = Path(sys._MEIPASS) / "jarvis" / "ui" / "static"
    else:
        static_dir = Path(__file__).parent / "static"
    index_path = static_dir / "index.html"
    icon_path = static_dir / "jarvis_icon.ico"

    index_url = _build_index_url(index_path, server.port)

    # Create and launch the window (blocks until closed)
    window = webview.create_window(
        title="J.A.R.V.I.S.",
        url=index_url,
        js_api=api,
        width=1200,
        height=800,
        min_size=(112, 112),
        frameless=True,
        on_top=False,
        transparent=True,
    )
    api._window = window

    # Route the configured global key into the same push-to-talk queue used by
    # the mic button.  It must not hide the window: the settings label promises
    # a voice override, and this keeps desktop and UI capture behaviour aligned.
    if voice_manager:
        try:
            from jarvis.voice.hotkey import HotkeyListener

            voice_cfg = session_manager.settings.voice
            hotkey = HotkeyListener(hotkey_combo=voice_cfg.push_to_talk_key)
            hotkey.start(server_loop, voice_manager.hotkey_queue)

            # Attach to server for dynamic rebinding
            server.hotkey_listener = hotkey
            server.hotkey_queue = voice_manager.hotkey_queue
            server.server_loop = server_loop
            asyncio.run_coroutine_threadsafe(voice_manager.start_ptt_loop(), server_loop)
        except ImportError:
            logger.warning("pynput not installed, global push-to-talk disabled")

    # Startup TTS Greeting (Slightly delayed to ensure audio init)
    async def delayed_greeting():
        await asyncio.sleep(0.5)
        if voice_manager:
            await voice_manager.tts.speak(
                "Systems loading. I am online and ready for your commands."
            )
        else:
            try:
                from jarvis.voice.tts import TTSProvider

                voice_cfg = session_manager.settings.voice
                tts = TTSProvider(
                    voice=voice_cfg.tts_voice,
                    rate=voice_cfg.tts_rate,
                    pitch=voice_cfg.tts_pitch,
                )
                await tts.speak("Systems loading. I am online and ready for your commands.")
            except Exception as e:
                logger.warning(f"Startup TTS failed: {e}")

    if voice_manager:
        asyncio.run_coroutine_threadsafe(delayed_greeting(), server_loop)

    # webview.start() blocks until the window is closed
    webview.start(gui="edgechromium", icon=str(icon_path.absolute()), debug=False)

    # Cleanup
    shutdown_event = startup_state.get("shutdown_event")
    if isinstance(shutdown_event, asyncio.Event):
        server_loop.call_soon_threadsafe(shutdown_event.set)
    else:
        server_loop.call_soon_threadsafe(server_loop.stop)
    server_thread.join(timeout=10)
    logger.info("JARVIS UI closed")
