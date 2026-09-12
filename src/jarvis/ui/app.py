"""JARVIS Desktop UI — pywebview window with WebSocket bridge."""

import asyncio
import threading
from pathlib import Path
from typing import Any

import structlog

from jarvis.ui.desktop import (
    WIDGET_SIZE,
    DesktopShortcuts,
    WindowGeometry,
    get_window_geometry,
    initial_window_size,
    restore_minimized_window,
    set_circular_region,
    set_click_through,
    set_window_geometry,
)
from jarvis.ui.server import WebSocketServer

logger = structlog.get_logger(__name__)


def _build_index_url(index_path: Path, port: int, token: str = "") -> str:
    """Build a WebView-safe file URL while passing runtime data in the fragment."""
    resolved = index_path.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"JARVIS UI entry point is not a file: {resolved}")
    # Edge/WebView2 may interpret a query string on file:// URLs as part of
    # the filename. URL fragments are never sent to the filesystem.
    return f"{resolved.as_uri()}#port={port}" + (f"&token={token}" if token else "")


def _exclude_hud_from_capture(window) -> None:
    """Let explicit screen requests see the application behind the assistant."""
    try:
        import ctypes
        from ctypes import wintypes

        hwnd = window.native.Handle.ToInt64()
        set_affinity = ctypes.windll.user32.SetWindowDisplayAffinity
        set_affinity.argtypes = [wintypes.HWND, wintypes.DWORD]
        set_affinity.restype = wintypes.BOOL
        if not set_affinity(hwnd, 0x11):
            logger.debug("HUD capture exclusion is not supported on this desktop")
    except Exception as exc:
        logger.debug("HUD capture exclusion unavailable", error=str(exc))


class JarvisAPI:
    """Python ↔ JavaScript bridge exposed to the webview window."""

    def __init__(self, session_manager, server_loop=None, server=None):
        self._session_manager = session_manager
        self._window: Any = None
        self._loop = server_loop
        self._server = server
        self._normal_geometry: WindowGeometry | None = None
        self._widget_position: tuple[int, int] | None = None
        self._normal_fullscreen = False
        self._widget_mode = False
        self._widget_interactive = False
        self._last_region_size: tuple[int, int] | None = None
        self._desktop_shortcuts: DesktopShortcuts | None = None
        self._window_lock = threading.RLock()

    def get_desktop_shortcuts(self) -> list[str]:
        from jarvis.ui.desktop import SHORTCUTS

        if not self._desktop_shortcuts:
            return []
        return [SHORTCUTS[key][0] for key in self._desktop_shortcuts.registered]

    def _shape_window(self, circular: bool) -> None:
        set_circular_region(self._window, circular)

    def toggle_widget_interaction(self) -> bool:
        with self._window_lock:
            if not self._widget_mode:
                return False
            self._widget_interactive = not self._widget_interactive
            set_click_through(self._window, not self._widget_interactive)
            return self._widget_interactive

    def _refresh_widget_shape(self, *args: Any) -> None:
        """Reclip after Windows rescales the orb on another monitor."""
        with self._window_lock:
            if self._widget_mode:
                try:
                    geometry = get_window_geometry(self._window)
                    size = (geometry.width, geometry.height)
                    if size != self._last_region_size:
                        self._shape_window(True)
                        self._last_region_size = size
                except Exception as exc:
                    logger.debug("Widget region refresh deferred", error=str(exc))

    def minimize_to_tray(self):
        """Minimize the window."""
        if self._window:
            self._window.minimize()

    def save_provider_keys(self, groq_key: str, gemini_key: str) -> dict:
        """Secrets use the native bridge and Windows encryption, never WebSocket events."""
        from jarvis.config.credentials import save_keys
        from jarvis.config.settings import get_settings

        try:
            save_keys(groq_key, gemini_key)
            get_settings.cache_clear()
            return {
                "ok": True,
                "message": "Keys saved securely. Close and reopen JARVIS to activate them.",
            }
        except Exception:
            return {
                "ok": False,
                "message": "Could not save keys. Enter valid comma-separated keys and retry.",
            }

    def collapse_to_widget(self) -> bool:
        """Collapse the window into a small floating widget."""
        if not self._window:
            return False
        with self._window_lock:
            if self._widget_mode:
                return True
            try:
                restore_minimized_window(self._window)
                self._normal_fullscreen = bool(self._window.fullscreen)
                if self._normal_fullscreen:
                    self._window.toggle_fullscreen()
                self._normal_geometry = get_window_geometry(self._window)
                normal = self._normal_geometry
                # Stay above pywebview's native minimum size. Resizing below
                # that boundary can terminate Edge/WebView2 on Windows.
                size = round(WIDGET_SIZE * normal.dpi / 96)
                x, y = self._widget_position or (normal.x, normal.y)
                set_window_geometry(
                    self._window,
                    WindowGeometry(x, y, size, size, True, normal.dpi, normal.visible),
                )
                # The remembered orb may be on a different DPI monitor. Move
                # first, then use that monitor's actual scale for its size.
                widget = get_window_geometry(self._window)
                target_size = round(WIDGET_SIZE * widget.dpi / 96)
                if widget.width != target_size or widget.height != target_size:
                    set_window_geometry(
                        self._window,
                        WindowGeometry(
                            x, y, target_size, target_size, True, widget.dpi, widget.visible
                        ),
                    )
                self._shape_window(True)
                set_click_through(self._window, False)
                self._widget_interactive = True
                self._widget_mode = True
                return True
            except Exception as exc:
                logger.error("Failed to enter widget mode", error=str(exc))
                try:
                    set_click_through(self._window, False)
                    self._shape_window(False)
                    if self._normal_geometry:
                        set_window_geometry(self._window, self._normal_geometry)
                    if self._normal_fullscreen and not self._window.fullscreen:
                        self._window.toggle_fullscreen()
                except Exception:
                    logger.warning("Failed to restore window after widget-mode error")
                return False

    def expand_to_window(self) -> bool:
        """Expand the window to full UI size."""
        if not self._window:
            return False
        with self._window_lock:
            if not self._widget_mode:
                return True
            widget_geometry: WindowGeometry | None = None
            try:
                restore_minimized_window(self._window)
                widget_geometry = get_window_geometry(self._window)
                self._widget_position = (widget_geometry.x, widget_geometry.y)
                set_click_through(self._window, False)
                self._shape_window(False)
                if self._normal_geometry:
                    set_window_geometry(self._window, self._normal_geometry)
                if self._normal_fullscreen and not self._window.fullscreen:
                    self._window.toggle_fullscreen()
                self._widget_mode = False
                return True
            except Exception as exc:
                logger.error("Failed to leave widget mode", error=str(exc))
                try:
                    if widget_geometry:
                        set_window_geometry(self._window, widget_geometry)
                    self._shape_window(True)
                    set_click_through(self._window, not self._widget_interactive)
                except Exception:
                    logger.warning("Failed to restore circle after expansion error")
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


def launch_ui(session_manager, voice_manager=None, *, smoke_report: Path | None = None):
    """Launch the JARVIS desktop UI.

    This creates a WebSocket server on a background thread
    and opens the pywebview window on the main thread.

    Args:
        session_manager: Initialized SessionManager instance.
        voice_manager: Optional initialized VoiceManager instance for Live Conversation.
    """
    import webview

    # Start WebSocket server on a background thread with its own event loop
    server_loop = asyncio.new_event_loop()
    server = WebSocketServer(session_manager, voice_manager=voice_manager)
    ready_event = threading.Event()
    startup_state: dict[str, object] = {}
    smoke_state: dict[str, Any] = {}
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
            raise RuntimeError(
                f"JARVIS backend initialization failed: {startup_error}"
            ) from startup_error
        raise RuntimeError("JARVIS backend initialization failed")

    # Session initialization configures structlog. A frozen windowed process
    # has no stdout, so logging before this point makes structlog's default
    # PrintLogger try to weak-reference ``None`` and terminates the app.
    logger.info("Launching JARVIS Desktop UI")

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

    index_url = _build_index_url(index_path, server.port, server.auth_token)
    initial_width, initial_height = initial_window_size()

    # Create and launch the window (blocks until closed)
    window = webview.create_window(
        title="J.A.R.V.I.S.",
        url=index_url,
        js_api=api,
        width=initial_width,
        height=initial_height,
        min_size=(112, 112),
        frameless=True,
        easy_drag=False,
        on_top=False,
        transparent=True,
        hidden=smoke_report is not None,
        focus=smoke_report is None,
    )
    api._window = window
    window.events.shown += lambda: _exclude_hud_from_capture(window)
    window.events.resized += api._refresh_widget_shape
    window.events.moved += api._refresh_widget_shape

    if smoke_report is None:

        def start_desktop_shortcuts():
            from jarvis.ui.desktop import DesktopShortcuts

            if api._desktop_shortcuts is None:
                api._desktop_shortcuts = DesktopShortcuts(window)
                api._desktop_shortcuts.start()
                window.evaluate_js("window.dispatchEvent(new Event('jarvis-shortcuts-ready'))")

        window.events.loaded += start_desktop_shortcuts

    if smoke_report is not None:

        def check_native_ui():
            import json
            import time

            from jarvis.ui.desktop import inspect_window_region

            result: dict = {"ok": False}
            try:
                # EdgeChromium's transparent navigation handler calls Show()
                # even when create_window(hidden=True) was requested. Once
                # navigation has loaded, hide it again before our baseline and
                # verify the real native transitions never show it afterward.
                window.hide()
                for _ in range(100):
                    result = window.evaluate_js("""({
                        ready: document.readyState,
                        connected: document.querySelector('.hud-sys-status').textContent,
                        bridge: !!(window.pywebview && window.pywebview.api),
                        screenButton: !!document.getElementById('screen-btn')
                    })""")
                    if result.get("bridge") and "CONNECTING" not in result.get(
                        "connected", "CONNECTING"
                    ):
                        result["ok"] = result.get("ready") == "complete" and result.get(
                            "screenButton"
                        )
                        break
                    time.sleep(0.1)
                if result.get("ok"):
                    original = get_window_geometry(window)
                    result["initialHidden"] = not original.visible
                    if original.visible:
                        raise RuntimeError("Could not hide the native verification window")
                    shortcuts = DesktopShortcuts(window)
                    if not shortcuts.dispatch("toggle"):
                        raise RuntimeError("Could not dispatch the widget shortcut")
                    for _ in range(100):
                        if api._widget_mode:
                            break
                        time.sleep(0.05)
                    time.sleep(1.0)
                    circle = inspect_window_region(window)
                    result["widget"] = circle
                    if not (
                        api._widget_mode
                        and circle["hasRegion"]
                        and circle["centerIncluded"]
                        and circle["cornersExcluded"]
                        and circle["width"] == circle["height"]
                        and not circle["clickThrough"]
                        and not circle["visible"]
                    ):
                        raise RuntimeError("Native circle geometry or hidden state is incorrect")
                    if (
                        api.toggle_widget_interaction()
                        or not inspect_window_region(window)["clickThrough"]
                    ):
                        raise RuntimeError("Optional click-through mode did not activate")
                    if (
                        not api.toggle_widget_interaction()
                        or inspect_window_region(window)["clickThrough"]
                    ):
                        raise RuntimeError("The orb did not become clickable again")
                    result["interactionToggle"] = True
                    # Allow the frontend's short transition to settle before
                    # testing the same global shortcut in the opposite direction.
                    time.sleep(0.4)
                    window.evaluate_js("document.getElementById('widget-expand-btn').click()")
                    for _ in range(100):
                        if not api._widget_mode:
                            break
                        time.sleep(0.05)
                    restored = get_window_geometry(window)
                    rectangle = inspect_window_region(window)
                    result["restored"] = rectangle
                    result["shortcutToggle"] = not api._widget_mode
                    if api._widget_mode or rectangle["hasRegion"] or restored != original:
                        raise RuntimeError("The HUD did not restore its original geometry")
            except Exception as exc:
                result.update({"ok": False, "error": str(exc)})
            finally:
                smoke_state.update(result)
                smoke_report.parent.mkdir(parents=True, exist_ok=True)
                smoke_report.write_text(json.dumps(result, indent=2), encoding="utf-8")
                window.destroy()

        window.events.loaded += check_native_ui

    # Route the configured global key into the same push-to-talk queue used by
    # the mic button.  It must not hide the window: the settings label promises
    # a voice override, and this keeps desktop and UI capture behaviour aligned.
    if voice_manager and smoke_report is None:
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
        except Exception as exc:
            logger.warning("Global hotkey unavailable; use the mic button", error=str(exc))

    # webview.start() blocks until the window is closed
    try:
        webview.start(gui="edgechromium", icon=str(icon_path.absolute()), debug=False)
    finally:
        if api._desktop_shortcuts:
            api._desktop_shortcuts.stop()
        shutdown_event = startup_state.get("shutdown_event")
        if isinstance(shutdown_event, asyncio.Event) and not server_loop.is_closed():
            server_loop.call_soon_threadsafe(shutdown_event.set)
        server_thread.join(timeout=10)

    logger.info("JARVIS UI closed")
    if smoke_report is not None and not smoke_state.get("ok"):
        raise RuntimeError(smoke_state.get("error", "Native UI smoke check did not complete"))
