"""Render and exercise the shipped HUD against a local test backend."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from playwright.async_api import async_playwright

from jarvis.config.settings import Settings
from jarvis.ui.app import _build_index_url
from jarvis.ui.server import WebSocketServer

ROOT = Path(__file__).resolve().parents[1]


async def main():
    requests = []

    async def response(text):
        requests.append(text)
        yield "The test reply is ready."

    session = SimpleNamespace(
        settings=Settings(groq_api_key="test", gemini_api_key="test"),
        router=SimpleNamespace(providers={"test": True}),
        memory=None,
        process_input_stream=response,
    )
    server = WebSocketServer(session, port=0)
    server._update_world_data_loop = AsyncMock()
    await server.start()
    errors = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(channel="msedge", headless=True)
            page = await browser.new_page(viewport={"width": 1200, "height": 800})
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.add_init_script("""window.nativeCalls = [];
                window.pywebview = {api: {
                    collapse_to_widget: async () => { nativeCalls.push('collapse'); return true; },
                    expand_to_window: async () => { nativeCalls.push('expand'); return true; },
                    toggle_widget_interaction: async () => !document.body.classList.contains('widget-interactive'),
                    get_desktop_shortcuts: async () => ['toggle', 'screen', 'stop', 'interact']
                }};""")
            await page.goto(
                _build_index_url(
                    ROOT / "src/jarvis/ui/static/index.html", server.port, server.auth_token
                )
            )
            await page.wait_for_function(
                "document.querySelector('.hud-sys-status').textContent.includes('CONNECTED')"
            )
            await page.evaluate("window.dispatchEvent(new Event('jarvis-shortcuts-ready'))")
            assert (
                await page.locator(".hud-message").first.evaluate(
                    "el => getComputedStyle(el).opacity"
                )
                == "1"
            )
            assert await page.locator("#shortcut-status").evaluate(
                "el => el.getBoundingClientRect().bottom <= innerHeight"
            )
            await page.screenshot(path=str(ROOT / "logs/ui-welcome.png"))
            await page.locator("#message-input").fill("Hello test")
            await page.locator("#send-btn").click()
            await page.get_by_text("The test reply is ready.", exact=False).wait_for()
            assert requests == ["Hello test"]
            await page.locator("#screen-btn").click()
            await page.wait_for_function(
                "document.querySelectorAll('.hud-message.jarvis').length === 2"
            )
            assert "screen" in requests[-1]
            await page.wait_for_function("!replyPending")
            await page.evaluate("""replyPending = true;
                handleMessage({type: 'notice', message: 'Selected voice temporarily unavailable'});""")
            assert await page.evaluate("replyPending")
            await page.evaluate("handleMessage({type: 'done'})")
            await page.locator("#message-input").fill("Keep my unfinished draft")
            await page.evaluate(
                "window.dispatchEvent(new CustomEvent('jarvis-shortcut', {detail: 'screen'}))"
            )
            await page.wait_for_function("!replyPending")
            assert await page.locator("#message-input").input_value() == "Keep my unfinished draft"
            await page.locator("#settings-btn").click()
            await page.locator("#groq-key").wait_for(state="visible")
            await page.locator("#setting-weather-city").fill("New")
            await page.locator("#setting-weather-city").press("Space")
            await page.locator("#setting-weather-city").press_sequentially("York")
            assert await page.locator("#setting-weather-city").input_value() == "New York"
            assert not await page.evaluate("isRecording")
            await page.locator("#save-settings-btn").scroll_into_view_if_needed()
            assert await page.locator("#save-settings-btn").is_visible()
            await page.screenshot(path=str(ROOT / "logs/ui-settings-smoke.png"))
            await page.locator("#close-settings-btn").click()
            await page.wait_for_function(
                "getComputedStyle(document.getElementById('settings-modal')).opacity === '0'"
            )
            await page.screenshot(path=str(ROOT / "logs/ui-smoke.png"))
            await page.evaluate("""window.voiceEvents = [];
                const originalSend = ws.send.bind(ws);
                ws.send = message => {
                    const type = JSON.parse(message).type;
                    if (type.startsWith('voice_')) voiceEvents.push(type);
                    else originalSend(message);
                };
                micBtn.disabled = false;
                widgetMicBtn.disabled = false;
            """)
            await page.locator("#mic-btn").focus()
            await page.keyboard.down("Space")
            assert await page.evaluate("isRecording")
            await page.keyboard.up("Space")
            assert await page.evaluate("voiceEvents") == ["voice_start", "voice_stop"]
            await page.evaluate("""updateCoreState('RECORDING');
                handleMessage({type: 'audio_level', level: .04});""")
            assert (
                await page.locator(".svg-center").evaluate("el => el.style.transform")
                == "scale(1.6)"
            )
            assert (
                await page.locator(".svg-center").evaluate(
                    "el => getComputedStyle(el).animationName"
                )
                == "none"
            )
            await page.evaluate("updateCoreState('IDLE')")
            await page.locator("#minimize-btn").click()
            await page.wait_for_function("isWidgetMode && !widgetTransition")
            await page.set_viewport_size({"width": 112, "height": 112})
            await page.mouse.move(0, 0)
            await page.screenshot(path=str(ROOT / "logs/ui-widget.png"), omit_background=True)
            assert await page.locator(".left-panel").is_hidden()
            assert (
                await page.evaluate(
                    "getComputedStyle(document.querySelector('.hud-container')).backgroundColor"
                )
                == "rgba(0, 0, 0, 0)"
            )
            assert await page.locator("#widget-expand-btn").is_visible()
            assert await page.evaluate(
                "document.elementFromPoint(56, 56).id === 'widget-expand-btn'"
            )
            await page.evaluate(
                "window.dispatchEvent(new CustomEvent('jarvis-shortcut', {detail: 'interact'}))"
            )
            await page.wait_for_function("!document.body.classList.contains('widget-interactive')")
            assert await page.locator("#widget-expand-btn").is_hidden()
            await page.evaluate(
                "window.dispatchEvent(new CustomEvent('jarvis-shortcut', {detail: 'interact'}))"
            )
            await page.wait_for_function("document.body.classList.contains('widget-interactive')")
            # Compact controls remain inside the native circular region.
            await page.locator(".ai-core-container").hover()
            await page.mouse.click(56, 56)
            await page.wait_for_function("!isWidgetMode && !widgetTransition")
            await page.set_viewport_size({"width": 1200, "height": 800})
            assert await page.locator("#message-input").input_value() == "Keep my unfinished draft"
            for _ in range(4):
                await page.evaluate(
                    "window.dispatchEvent(new CustomEvent('jarvis-shortcut', {detail: 'toggle'}))"
                )
                await page.wait_for_function("!widgetTransition")
            assert await page.evaluate("nativeCalls") == ["collapse", "expand"] * 3
            await page.set_viewport_size({"width": 850, "height": 600})
            await page.screenshot(path=str(ROOT / "logs/ui-compact-window.png"))
            assert await page.locator("#message-input").is_visible()
            assert await page.locator("#minimize-btn").is_visible()
            assert await page.locator("#shortcut-status").evaluate(
                "el => el.getBoundingClientRect().bottom <= innerHeight"
            )
            await page.emulate_media(reduced_motion="reduce")
            assert (
                await page.locator(".hud-message").first.evaluate(
                    "el => getComputedStyle(el).opacity"
                )
                == "1"
            )
            assert not errors, errors
            await browser.close()
    finally:
        await server.stop()
    (ROOT / "logs/ui-smoke.json").write_text(
        json.dumps(
            {
                "ok": True,
                "checks": [
                    "connect",
                    "chat",
                    "screen_button",
                    "draft_preserved",
                    "settings_typing",
                    "keyboard_microphone",
                    "audio_reactive_core",
                    "circular_widget",
                    "restore",
                    "repeat_shortcuts",
                    "compact_layout",
                    "reduced_motion",
                    "no_js_errors",
                ],
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
