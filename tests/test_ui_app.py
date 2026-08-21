"""Tests for desktop UI startup path handling."""

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from bs4 import BeautifulSoup

from jarvis.ui.app import JarvisAPI, _build_index_url, _run_server


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


def test_widget_mode_uses_safe_native_size_and_restores_window():
    class FakeWindow:
        width = 1400
        height = 900
        on_top = False

        def __init__(self):
            self.sizes: list[tuple[int, int]] = []

        def resize(self, width: int, height: int) -> None:
            self.sizes.append((width, height))

    window = FakeWindow()
    api = JarvisAPI(SimpleNamespace())
    api._window = window

    assert api.collapse_to_widget() is True
    assert window.sizes[-1] == (112, 112)
    assert window.on_top is True

    assert api.expand_to_window() is True
    assert window.sizes[-1] == (1400, 900)
    assert window.on_top is False


def test_widget_resize_failure_is_contained():
    class BrokenWindow:
        width = 1200
        height = 800
        on_top = False

        def resize(self, width: int, height: int) -> None:
            raise RuntimeError("native resize failed")

    api = JarvisAPI(SimpleNamespace())
    api._window = BrokenWindow()

    assert api.collapse_to_widget() is False


def test_world_monitor_is_a_full_hud_overlay():
    static_dir = Path(__file__).parents[1] / "src" / "jarvis" / "ui" / "static"
    soup = BeautifulSoup((static_dir / "index.html").read_text(encoding="utf-8"), "html.parser")
    monitor = soup.find(id="world-monitor")

    assert monitor is not None
    assert "hud-container" in monitor.parent.get("class", [])
    assert monitor.find_parent(class_="left-panel") is None

    styles = (static_dir / "styles.css").read_text(encoding="utf-8")
    assert "inset: 58px 15px 15px" in styles
    assert styles.count("body.widget-mode .dashboard-grid {") == 1


def test_widget_uses_separate_drag_surface_and_restore_button():
    static_dir = Path(__file__).parents[1] / "src" / "jarvis" / "ui" / "static"
    html = (static_dir / "index.html").read_text(encoding="utf-8")
    script = (static_dir / "app.js").read_text(encoding="utf-8")
    styles = (static_dir / "styles.css").read_text(encoding="utf-8")

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
