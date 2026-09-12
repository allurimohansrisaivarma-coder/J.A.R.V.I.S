"""Windows circular window geometry and focus-independent desktop shortcuts."""

import ctypes
import json
import threading
from ctypes import wintypes
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import structlog

logger = structlog.get_logger(__name__)
SHORTCUTS = {1: ("toggle", "J"), 2: ("screen", "R"), 3: ("stop", "X"), 4: ("interact", "D")}
WIDGET_SIZE = 112


@lru_cache(maxsize=1)
def _win32() -> tuple[Any, Any]:
    """Declare pointer-sized handles explicitly for 64-bit Windows."""
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetWindowRect.restype = wintypes.BOOL
    user32.SetWindowPos.argtypes = (
        [wintypes.HWND, wintypes.HWND] + [ctypes.c_int] * 4 + [wintypes.UINT]
    )
    user32.SetWindowPos.restype = wintypes.BOOL
    user32.SetWindowRgn.argtypes = [wintypes.HWND, wintypes.HRGN, wintypes.BOOL]
    user32.GetWindowRgn.argtypes = [wintypes.HWND, wintypes.HRGN]
    user32.GetDpiForWindow.argtypes = [wintypes.HWND]
    user32.GetDpiForWindow.restype = wintypes.UINT
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
    user32.SetLayeredWindowAttributes.argtypes = [
        wintypes.HWND,
        wintypes.DWORD,
        wintypes.BYTE,
        wintypes.DWORD,
    ]
    user32.SetThreadDpiAwarenessContext.argtypes = [wintypes.HANDLE]
    user32.SetThreadDpiAwarenessContext.restype = wintypes.HANDLE
    user32.SystemParametersInfoW.argtypes = [
        wintypes.UINT,
        wintypes.UINT,
        wintypes.LPVOID,
        wintypes.UINT,
    ]
    for name in ("CreateEllipticRgn", "CreateRectRgn"):
        function = getattr(gdi32, name)
        function.argtypes = [ctypes.c_int] * 4
        function.restype = wintypes.HRGN
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.PtInRegion.argtypes = [wintypes.HRGN, ctypes.c_int, ctypes.c_int]
    gdi32.PtInRegion.restype = wintypes.BOOL
    user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
    user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.GetMessageW.argtypes = [
        ctypes.POINTER(wintypes.MSG),
        wintypes.HWND,
        wintypes.UINT,
        wintypes.UINT,
    ]
    user32.PeekMessageW.argtypes = [
        ctypes.POINTER(wintypes.MSG),
        wintypes.HWND,
        wintypes.UINT,
        wintypes.UINT,
        wintypes.UINT,
    ]
    user32.PostThreadMessageW.argtypes = [
        wintypes.DWORD,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    ]
    return user32, gdi32


def fit_window_size(work_width: int, work_height: int) -> tuple[int, int]:
    """Leave room around the HUD without exceeding a smaller desktop."""
    return min(1200, max(WIDGET_SIZE, work_width - 48)), min(
        800, max(WIDGET_SIZE, work_height - 48)
    )


def initial_window_size() -> tuple[int, int]:
    """Get the primary work area in logical pixels, excluding the taskbar."""
    try:
        user32, _ = _win32()
        # pywebview expects logical size. Asking from an unaware context makes
        # this consistent whether a source process or an EXE is already DPI aware.
        previous = user32.SetThreadDpiAwarenessContext(-1)
        try:
            bounds = wintypes.RECT()
            if not user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(bounds), 0):
                raise ctypes.WinError(ctypes.get_last_error())
            return fit_window_size(bounds.right - bounds.left, bounds.bottom - bounds.top)
        finally:
            if previous:
                user32.SetThreadDpiAwarenessContext(previous)
    except Exception as exc:
        logger.warning("Desktop work area unavailable; using compact startup size", error=str(exc))
        return 1024, 640


@dataclass(frozen=True)
class WindowGeometry:
    """Physical desktop coordinates, so positions survive different monitor DPIs."""

    x: int
    y: int
    width: int
    height: int
    on_top: bool = False
    dpi: int = 96
    visible: bool = True


def get_window_geometry(window: Any) -> WindowGeometry:
    user32, _ = _win32()
    hwnd = window.native.Handle.ToInt64()
    bounds = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(bounds)):
        raise ctypes.WinError(ctypes.get_last_error())
    return WindowGeometry(
        bounds.left,
        bounds.top,
        bounds.right - bounds.left,
        bounds.bottom - bounds.top,
        bool(user32.GetWindowLongW(hwnd, -20) & 0x00000008),
        user32.GetDpiForWindow(hwnd) or 96,
        bool(user32.IsWindowVisible(hwnd)),
    )


def restore_minimized_window(window: Any) -> None:
    """A toggle can restore a minimized HUD without taking keyboard focus."""
    user32, _ = _win32()
    hwnd = window.native.Handle.ToInt64()
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 4)  # SW_SHOWNOACTIVATE; does not show hidden test windows.


def set_window_geometry(window: Any, geometry: WindowGeometry) -> None:
    """Resize and move without pywebview's unconditional SHOWWINDOW/activation."""
    user32, _ = _win32()
    hwnd = window.native.Handle.ToInt64()
    if not user32.SetWindowPos(
        hwnd,
        -1 if geometry.on_top else -2,  # HWND_TOPMOST / HWND_NOTOPMOST
        geometry.x,
        geometry.y,
        geometry.width,
        geometry.height,
        0x0010 | 0x0200,  # SWP_NOACTIVATE | SWP_NOOWNERZORDER
    ):
        raise ctypes.WinError(ctypes.get_last_error())


def set_circular_region(window: Any, circular: bool) -> None:
    """Clip rendering AND mouse hit testing using the current physical size."""
    user32, gdi32 = _win32()
    hwnd = window.native.Handle.ToInt64()
    region = None
    if circular:
        geometry = get_window_geometry(window)
        region = gdi32.CreateEllipticRgn(0, 0, geometry.width, geometry.height)
        if not region:
            raise ctypes.WinError(ctypes.get_last_error())
    if not user32.SetWindowRgn(hwnd, region, True):
        if region:
            gdi32.DeleteObject(region)
        raise ctypes.WinError(ctypes.get_last_error())
    # After success Windows owns the region; do not delete it.


def set_click_through(window: Any, enabled: bool) -> None:
    """Let all pointer input reach the app under the orb, not just its corners."""
    user32, _ = _win32()
    hwnd = window.native.Handle.ToInt64()
    style = user32.GetWindowLongW(hwnd, -20)
    if enabled:
        user32.SetWindowLongW(hwnd, -20, style | 0x80000 | 0x20)
        if not user32.SetLayeredWindowAttributes(hwnd, 0, 255, 2):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        user32.SetWindowLongW(hwnd, -20, style & ~0x20)


def inspect_window_region(window: Any) -> dict[str, Any]:
    """Read the actual OS clipping region for native release verification."""
    user32, gdi32 = _win32()
    geometry = get_window_geometry(window)
    region = gdi32.CreateRectRgn(0, 0, 0, 0)
    if not region:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        region_type = user32.GetWindowRgn(window.native.Handle.ToInt64(), region)
        corners = [
            (0, 0),
            (geometry.width - 1, 0),
            (0, geometry.height - 1),
            (geometry.width - 1, geometry.height - 1),
        ]
        return {
            "hasRegion": bool(region_type),
            "centerIncluded": bool(
                gdi32.PtInRegion(region, geometry.width // 2, geometry.height // 2)
            ),
            "cornersExcluded": all(not gdi32.PtInRegion(region, x, y) for x, y in corners),
            "width": geometry.width,
            "height": geometry.height,
            "visible": geometry.visible,
            "clickThrough": bool(user32.GetWindowLongW(window.native.Handle.ToInt64(), -20) & 0x20),
        }
    finally:
        gdi32.DeleteObject(region)


class DesktopShortcuts:
    """Register system shortcuts without keyboard polling or stealing focus."""

    def __init__(self, window: Any):
        self.window = window
        self.thread_id = 0
        self.thread: threading.Thread | None = None
        self.ready = threading.Event()
        self.stopping = threading.Event()
        self.registered: list[int] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self.thread and self.thread.is_alive():
                return
            self.ready.clear()
            self.stopping.clear()
            self.thread = threading.Thread(target=self._run, daemon=True, name="jarvis-shortcuts")
            self.thread.start()
        self.ready.wait(3)

    def dispatch(self, action: str) -> bool:
        """Send an action to the HUD without activating or raising its window."""
        if action not in {value[0] for value in SHORTCUTS.values()} or self.stopping.is_set():
            return False
        try:
            self.window.evaluate_js(
                "window.dispatchEvent(new CustomEvent('jarvis-shortcut',"
                f"{{detail:{json.dumps(action)}}}))"
            )
            return True
        except Exception:
            logger.debug("Desktop shortcut ignored while window closes")
            return False

    def _run(self) -> None:
        user32, _ = _win32()
        message = wintypes.MSG()
        try:
            self.thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
            # Ensure PostThreadMessage can stop us even if every key is in use.
            user32.PeekMessageW(ctypes.byref(message), None, 0, 0, 0)
            for identifier, (_, key) in SHORTCUTS.items():
                if self.stopping.is_set():
                    break
                # MOD_NOREPEAT prevents a held key from repeatedly resizing.
                if user32.RegisterHotKey(None, identifier, 0x4003, ord(key)):
                    self.registered.append(identifier)
                else:
                    logger.warning("Desktop shortcut already in use", shortcut=f"Ctrl+Alt+{key}")
            self.ready.set()
            while not self.stopping.is_set():
                status = user32.GetMessageW(ctypes.byref(message), None, 0, 0)
                if status <= 0:
                    break
                if message.message == 0x0312 and message.wParam in self.registered:
                    self.dispatch(SHORTCUTS[message.wParam][0])
        except Exception as exc:
            logger.warning("Desktop shortcuts unavailable", error=str(exc))
        finally:
            for identifier in self.registered:
                user32.UnregisterHotKey(None, identifier)
            self.registered.clear()
            self.thread_id = 0
            self.ready.set()

    def stop(self) -> None:
        self.stopping.set()
        if self.thread_id:
            user32, _ = _win32()
            user32.PostThreadMessageW(self.thread_id, 0x0012, 0, 0)
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=3)
