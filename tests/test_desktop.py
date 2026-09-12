"""Native geometry and shortcut ownership regressions."""

import ctypes
import sys
from ctypes import wintypes
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jarvis.ui import desktop


@pytest.mark.parametrize(
    "work_area,expected",
    [((1280, 672), (1200, 624)), ((1920, 1040), (1200, 800)), ((800, 600), (752, 552))],
)
def test_initial_hud_fits_logical_work_area(work_area, expected):
    assert desktop.fit_window_size(*work_area) == expected


def test_screen_shortcut_does_not_activate_or_resize_the_hud():
    window = SimpleNamespace(evaluate_js=Mock())
    shortcuts = desktop.DesktopShortcuts(window)
    assert shortcuts.dispatch("screen")
    script = window.evaluate_js.call_args.args[0]
    assert "jarvis-shortcut" in script
    assert 'detail:"screen"' in script
    assert not shortcuts.dispatch("unrecognized")
    shortcuts.stop()
    assert not shortcuts.dispatch("screen")
    window.evaluate_js.assert_called_once()


def test_shortcut_loop_unregisters_only_successful_registrations(monkeypatch):
    window = SimpleNamespace(evaluate_js=Mock())
    calls = 0

    def get_message(pointer, *args):
        nonlocal calls
        calls += 1
        if calls > 1:
            return 0
        message = ctypes.cast(pointer, ctypes.POINTER(wintypes.MSG)).contents
        message.message = 0x0312
        message.wParam = 1
        return 1

    user32 = SimpleNamespace(
        PeekMessageW=Mock(),
        RegisterHotKey=Mock(side_effect=[True, False, True, False]),
        UnregisterHotKey=Mock(),
        GetMessageW=Mock(side_effect=get_message),
    )
    monkeypatch.setattr(desktop, "_win32", lambda: (user32, None))
    shortcuts = desktop.DesktopShortcuts(window)
    shortcuts._run()
    assert shortcuts.ready.is_set()
    assert shortcuts.registered == []
    assert shortcuts.thread_id == 0
    assert [call.args[1] for call in user32.UnregisterHotKey.call_args_list] == [1, 3]
    assert all(call.args[2] == 0x4003 for call in user32.RegisterHotKey.call_args_list)
    assert 'detail:"toggle"' in window.evaluate_js.call_args.args[0]
    user32.PeekMessageW.assert_called_once()


@pytest.mark.skipif(sys.platform != "win32", reason="Native Windows geometry")
def test_real_native_circle_excludes_corners_and_stays_hidden():
    """Use a tiny hidden system window; never touch the user's applications."""
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.HWND,
        wintypes.HMENU,
        wintypes.HINSTANCE,
        wintypes.LPVOID,
    ]
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    hwnd = user32.CreateWindowExW(
        0,
        "Static",
        "JARVIS geometry verification",
        0x80000000,
        20,
        20,
        168,
        168,
        None,
        None,
        None,
        None,
    )
    assert hwnd, ctypes.WinError(ctypes.get_last_error())
    window = SimpleNamespace(native=SimpleNamespace(Handle=SimpleNamespace(ToInt64=lambda: hwnd)))
    try:
        original = desktop.get_window_geometry(window)
        assert not original.visible
        desktop.set_circular_region(window, True)
        circle = desktop.inspect_window_region(window)
        assert circle["hasRegion"]
        assert circle["centerIncluded"]
        assert circle["cornersExcluded"]
        assert circle["width"] == circle["height"]
        assert not circle["visible"]
        desktop.set_click_through(window, True)
        assert desktop.inspect_window_region(window)["clickThrough"]
        desktop.set_click_through(window, False)
        assert not desktop.inspect_window_region(window)["clickThrough"]
        desktop.set_circular_region(window, False)
        desktop.set_window_geometry(window, desktop.WindowGeometry(40, 40, 320, 240, True))
        restored = desktop.inspect_window_region(window)
        assert not restored["hasRegion"]
        assert restored["width"] == 320
        assert restored["height"] == 240
        assert not restored["visible"]
        assert desktop.get_window_geometry(window).on_top
        desktop.set_window_geometry(window, original)
        assert desktop.get_window_geometry(window) == original
    finally:
        user32.DestroyWindow(hwnd)
