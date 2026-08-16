"""JARVIS Desktop UI — pywebview window with WebSocket bridge."""

import threading
import asyncio
from pathlib import Path
import structlog

from jarvis.ui.server import WebSocketServer

logger = structlog.get_logger(__name__)


class JarvisAPI:
    """Python ↔ JavaScript bridge exposed to the webview window."""
    
    def __init__(self, session_manager, server_loop=None, server=None):
        self._session_manager = session_manager
        self._window = None
        self._voice_active = False
        self._loop = server_loop
        self._server = server

    def send_message(self, text: str):
        """Send a chat message (routed via WebSocket instead)."""
        logger.info("Message received via API", text=text)

    def toggle_voice(self):
        """Toggle voice listening on/off."""
        self._voice_active = not self._voice_active
        state = "LISTENING" if self._voice_active else "IDLE"
        logger.info("Voice toggled via API", state=state)
        if self._server and self._loop:
            asyncio.run_coroutine_threadsafe(
                self._server.broadcast_state(state),
                self._loop
            )
        return self._voice_active

    def get_status(self) -> dict:
        """Return current system status."""
        return {"status": "OK", "voice_active": self._voice_active}

    def minimize_to_tray(self):
        """Minimize the window."""
        if self._window:
            self._window.minimize()

    def collapse_to_widget(self):
        """Collapse the window into a small floating widget."""
        if self._window:
            self._window.resize(80, 80)

    def expand_to_window(self):
        """Expand the window to full UI size."""
        if self._window:
            self._window.resize(420, 650)


def _run_server(server, loop):
    """Run the WebSocket server in its own event loop on a background thread."""
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.start())
    try:
        loop.run_forever()
    finally:
        loop.run_until_complete(server.stop())
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
    server_thread = threading.Thread(
        target=_run_server, 
        args=(server, server_loop), 
        daemon=True
    )
    server_thread.start()

    # Create the API bridge
    api = JarvisAPI(session_manager)
    api._loop = server_loop
    api._server = server

    # Resolve the static files path
    import sys
    if hasattr(sys, '_MEIPASS'):
        static_dir = Path(sys._MEIPASS) / 'jarvis' / 'ui' / 'static'
    else:
        static_dir = Path(__file__).parent / 'static'
    index_path = static_dir / 'index.html'
    icon_path = static_dir / 'jarvis_icon.ico'
    
    # On Windows, file:// URLs need forward slashes and proper formatting
    index_url = index_path.absolute().as_uri()

    # Create and launch the window (blocks until closed)
    window = webview.create_window(
        title='J.A.R.V.I.S.',
        url=index_url,
        js_api=api,
        width=420,
        height=650,
        frameless=True,
        on_top=True,
        transparent=False,
        background_color='#05080f'
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

    # Startup TTS Greeting
    if voice_manager:
        asyncio.run_coroutine_threadsafe(voice_manager.tts.speak("System Starting"), server_loop)

    # webview.start() blocks until the window is closed
    webview.start(gui='edgechromium', icon=str(icon_path.absolute()), debug=True)
    
    # Cleanup
    server_loop.call_soon_threadsafe(server_loop.stop)
    server_thread.join(timeout=2)
    logger.info("JARVIS UI closed")
