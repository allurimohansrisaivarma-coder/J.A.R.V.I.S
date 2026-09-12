"""Tests for desktop UI startup path handling."""

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from bs4 import BeautifulSoup

from jarvis.ui.app import JarvisAPI, _build_index_url, _run_server
from jarvis.ui.desktop import WindowGeometry


def test_index_url_uses_fragment_for_runtime_port(tmp_path: Path):
    index = tmp_path / "UI files" / "index.html"
    index.parent.mkdir()
    index.write_text("<html></html>", encoding="utf-8")

    url = _build_index_url(index, 8741)

    assert url.startswith("file:///")
    assert "UI%20files/index.html#port=8741" in url
    assert "?port=" not in url


def test_index_url_fails_early_when_entry_point_is_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        _build_index_url(tmp_path / "missing.html", 8741)


@pytest.fixture
def native_window(monkeypatch):
    window = SimpleNamespace(
        geometry=WindowGeometry(200, 100, 1400, 900), fullscreen=False, circular=False
    )

    def set_geometry(target, geometry):
        target.geometry = geometry

    def set_shape(target, circular):
        target.circular = circular

    monkeypatch.setattr("jarvis.ui.app.get_window_geometry", lambda target: target.geometry)
    monkeypatch.setattr("jarvis.ui.app.set_window_geometry", set_geometry)
    monkeypatch.setattr("jarvis.ui.app.set_circular_region", set_shape)
    monkeypatch.setattr(
        "jarvis.ui.app.set_click_through",
        lambda target, enabled: setattr(target, "click_through", enabled),
    )
    monkeypatch.setattr("jarvis.ui.app.restore_minimized_window", lambda target: None)
    return window


def test_widget_mode_uses_safe_native_size_and_restores_window(native_window):
    window = native_window
    original = window.geometry
    api = JarvisAPI(SimpleNamespace())
    api._window = window

    assert api.collapse_to_widget() is True
    assert window.geometry == WindowGeometry(200, 100, 112, 112, True)
    assert window.circular is True
    assert window.click_through is False
    assert api.toggle_widget_interaction() is False
    assert window.click_through is True
    assert api.toggle_widget_interaction() is True
    assert window.click_through is False
    # Duplicate native requests must not replace the remembered main geometry.
    assert api.collapse_to_widget() is True

    assert api.expand_to_window() is True
    assert window.geometry == original
    assert window.circular is False
    assert window.click_through is False
    assert api.expand_to_window() is True
    assert window.geometry == original


def test_widget_remembers_its_position_independently(native_window):
    api = JarvisAPI(SimpleNamespace())
    api._window = native_window
    main = native_window.geometry
    assert api.collapse_to_widget()
    dragged = WindowGeometry(1800, 850, 112, 112, True)
    native_window.geometry = dragged
    assert api.expand_to_window()
    assert native_window.geometry == main
    assert api.collapse_to_widget()
    assert native_window.geometry == dragged


def test_widget_size_uses_monitor_dpi_and_keeps_hidden_window_hidden(native_window):
    native_window.geometry = WindowGeometry(-1600, 300, 1800, 1200, dpi=144, visible=False)
    api = JarvisAPI(SimpleNamespace())
    api._window = native_window
    assert api.collapse_to_widget()
    assert native_window.geometry.width == native_window.geometry.height == 168
    assert native_window.geometry.visible is False
    assert api.expand_to_window()
    assert native_window.geometry.visible is False


def test_widget_resize_failure_is_contained(native_window, monkeypatch):
    original = native_window.geometry
    real_setter = __import__("jarvis.ui.app", fromlist=["set_window_geometry"]).set_window_geometry

    def fail_collapse(target, geometry):
        if geometry.width == 112:
            raise RuntimeError("native resize failed")
        real_setter(target, geometry)

    monkeypatch.setattr("jarvis.ui.app.set_window_geometry", fail_collapse)
    api = JarvisAPI(SimpleNamespace())
    api._window = native_window

    assert api.collapse_to_widget() is False
    assert not api._widget_mode
    assert native_window.geometry == original
    assert not native_window.circular


def test_failed_expansion_keeps_a_usable_circle(native_window, monkeypatch):
    api = JarvisAPI(SimpleNamespace())
    api._window = native_window
    assert api.collapse_to_widget()
    widget = native_window.geometry
    setter = Mock(side_effect=[RuntimeError("native resize failed"), None])
    monkeypatch.setattr("jarvis.ui.app.set_window_geometry", setter)
    assert api.expand_to_window() is False
    assert api._widget_mode
    assert native_window.circular
    assert setter.call_args.args == (native_window, widget)


def test_world_monitor_is_a_full_hud_overlay():
    static_dir = Path(__file__).parents[1] / "src" / "jarvis" / "ui" / "static"
    soup = BeautifulSoup((static_dir / "index.html").read_text(encoding="utf-8"), "html.parser")
    monitor = soup.find(id="world-monitor")

    assert monitor is not None
    assert "hud-container" in monitor.parent.get("class", [])
    assert monitor.find_parent(class_="left-panel") is None

    styles = "\n".join(path.read_text(encoding="utf-8") for path in static_dir.glob("*.css"))
    assert "inset: 58px 15px 15px" in styles
    assert "body.widget-mode .dashboard-grid" in styles


def test_widget_uses_separate_drag_surface_and_restore_button():
    static_dir = Path(__file__).parents[1] / "src" / "jarvis" / "ui" / "static"
    html = (static_dir / "index.html").read_text(encoding="utf-8")
    script = (static_dir / "app.js").read_text(encoding="utf-8")
    styles = "\n".join(path.read_text(encoding="utf-8") for path in static_dir.glob("*.css"))

    assert 'id="widget-expand-btn"' in html
    assert 'widgetExpandBtn.addEventListener("click"' in script
    assert 'aiCore.addEventListener("click"' not in script
    assert "body.widget-mode .ai-core {" in styles
    assert "pointer-events: none;" in styles


def test_backend_opens_and_closes_resources_in_same_async_task():
    task_ids: list[int] = []

    class FakeSession:
        settings = SimpleNamespace(
            system=SimpleNamespace(port=8741),
            voice=SimpleNamespace(enabled=False),
        )

        async def initialize(self):
            task_ids.append(id(asyncio.current_task()))

        async def shutdown(self):
            task_ids.append(id(asyncio.current_task()))

    class FakeServer:
        session_manager = FakeSession()
        voice_manager = None
        port = 0

        async def start(self):
            task_ids.append(id(asyncio.current_task()))

        async def stop(self):
            task_ids.append(id(asyncio.current_task()))

    loop = asyncio.new_event_loop()
    ready = threading.Event()
    state: dict[str, object] = {}
    thread = threading.Thread(target=_run_server, args=(FakeServer(), loop, ready, state))
    thread.start()

    assert ready.wait(timeout=2)
    shutdown_event = state["shutdown_event"]
    assert isinstance(shutdown_event, asyncio.Event)
    loop.call_soon_threadsafe(shutdown_event.set)
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert len(set(task_ids)) == 1
