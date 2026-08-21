"""Global hotkey listener for push-to-talk control."""

import asyncio
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class HotkeyListener:
    """Listens for global keyboard shortcuts to control push-to-talk.

    Uses pynput to capture key events from a background thread
    and bridges them to the asyncio event loop via a Queue.
    """

    def __init__(self, hotkey_combo: str = "ctrl+shift+j"):
        """Initialize the hotkey listener.

        Args:
            hotkey_combo: Key combination string (e.g., 'ctrl+shift+j', 'f8').
        """
        self.hotkey_combo = hotkey_combo.lower()
        self._listener: Any | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[str] | None = None
        self._pressed_keys: set = set()
        self._hotkey_active = False
        self._required_keys = self._parse_combo(hotkey_combo)

    def _parse_combo(self, combo: str) -> set:
        """Parse a hotkey combo string into a set of key identifiers."""
        parts = [p.strip().lower() for p in combo.split("+")]
        keys = set()
        for part in parts:
            if part in ("ctrl", "control"):
                keys.add("ctrl")
            elif part in ("alt", "menu"):
                keys.add("alt")
            elif part in ("shift",):
                keys.add("shift")
            elif part in ("cmd", "win", "super", "command"):
                keys.add("cmd")
            else:
                keys.add(part)
        return keys

    def _key_to_id(self, key) -> str | None:
        """Convert a pynput key to our internal identifier."""
        from pynput import keyboard

        if isinstance(key, keyboard.Key):
            key_map = {
                keyboard.Key.ctrl_l: "ctrl",
                keyboard.Key.ctrl_r: "ctrl",
                keyboard.Key.alt_l: "alt",
                keyboard.Key.alt_r: "alt",
                keyboard.Key.alt_gr: "alt",
                keyboard.Key.shift_l: "shift",
                keyboard.Key.shift_r: "shift",
                keyboard.Key.cmd: "cmd",
                keyboard.Key.cmd_r: "cmd",
                keyboard.Key.f1: "f1",
                keyboard.Key.f2: "f2",
                keyboard.Key.f3: "f3",
                keyboard.Key.f4: "f4",
                keyboard.Key.f5: "f5",
                keyboard.Key.f6: "f6",
                keyboard.Key.f7: "f7",
                keyboard.Key.f8: "f8",
                keyboard.Key.f9: "f9",
                keyboard.Key.f10: "f10",
                keyboard.Key.f11: "f11",
                keyboard.Key.f12: "f12",
            }
            return key_map.get(key)
        elif hasattr(key, "vk") and key.vk is not None:
            # Prioritize vk because holding Ctrl can change key.char to a control character (e.g. \\x0a for J)
            if 65 <= key.vk <= 90 or 97 <= key.vk <= 122:
                return chr(key.vk).lower()
            elif hasattr(key, "char") and key.char:
                return key.char.lower()
        elif hasattr(key, "char") and key.char:
            return key.char.lower()
        return None

    def _on_press(self, key):
        """Handle key press events (runs in pynput's thread)."""
        key_id = self._key_to_id(key)
        if key_id is None:
            return

        self._pressed_keys.add(key_id)

        # Check if all required keys are currently pressed
        if self._required_keys.issubset(self._pressed_keys) and not self._hotkey_active:
            self._hotkey_active = True
            if self._loop and self._queue:
                self._loop.call_soon_threadsafe(self._queue.put_nowait, "PTT_START")
                logger.debug("Hotkey pressed", combo=self.hotkey_combo)

    def _on_release(self, key):
        """Handle key release events (runs in pynput's thread)."""
        key_id = self._key_to_id(key)
        if key_id is None:
            return

        self._pressed_keys.discard(key_id)

        # If any required key is released, trigger stop
        if self._hotkey_active and not self._required_keys.issubset(self._pressed_keys):
            self._hotkey_active = False
            if self._loop and self._queue:
                self._loop.call_soon_threadsafe(self._queue.put_nowait, "PTT_STOP")
                logger.debug("Hotkey released", combo=self.hotkey_combo)

    def start(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[str]):
        """Start listening for the global hotkey.

        Args:
            loop: The asyncio event loop to bridge events into.
            queue: The queue to put PTT_START/PTT_STOP events into.
        """
        from pynput import keyboard

        self._loop = loop
        self._queue = queue

        listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        listener.daemon = True
        listener.start()
        self._listener = listener
        logger.info("Global hotkey listener started", combo=self.hotkey_combo)

    def stop(self):
        """Stop the hotkey listener."""
        if self._listener:
            self._listener.stop()
            self._listener = None
        logger.info("Global hotkey listener stopped")
