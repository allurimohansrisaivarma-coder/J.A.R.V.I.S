"""Regressions reproduced from the shipped executable and its failure paths."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import websockets
from PIL import Image

from jarvis.config.settings import RouterSettings, Settings
from jarvis.core.conversation import ConversationManager
from jarvis.llm.base import LLMError, Message, ModelTier
from jarvis.llm.groq_provider import GroqProvider
from jarvis.llm.router import ModelRouter
from jarvis.tools.mcp_manager import MCPManager
from jarvis.ui.server import WebSocketServer
from jarvis.voice.stt import STTProvider
from jarvis.voice.tts import TTSProvider


def test_api_keys_are_encrypted_and_partial_update_keeps_other_provider(tmp_path, monkeypatch):
    import jarvis.config.settings as config
    from jarvis.config.credentials import load_keys, save_keys

    monkeypatch.setattr(config, "USER_DIR", tmp_path)
    save_keys("groq-test-secret", "gemini-test-secret")
    blob = (tmp_path / "provider-keys.dat").read_bytes()
    assert b"groq-test-secret" not in blob
    save_keys("new-groq-key", "")
    assert load_keys() == {
        "groq_api_keys": ["new-groq-key"],
        "gemini_api_keys": ["gemini-test-secret"],
    }


def test_push_to_talk_does_not_load_unused_vad(monkeypatch):
    from jarvis.voice import capture

    model = MagicMock(side_effect=RuntimeError("VAD model missing"))
    monkeypatch.setattr(capture, "SileroVAD", model)
    recorder = capture.AudioCapture()
    assert recorder._vad is None
    model.assert_not_called()


def test_request_images_do_not_pollute_history():
    conversation = ConversationManager()
    conversation.add_user_message("Read my screen")
    messages = conversation.get_context_messages()
    messages[-1].content = [messages[-1].content, Image.new("RGB", (20, 20))]
    assert conversation.get_context_messages()[-1].content == "Read my screen"


def test_groq_vision_sends_pixels_without_resizing_original():
    provider = GroqProvider(api_key="test", supports_images=True)
    original = Image.new("RGBA", (2200, 1600))
    payload = provider._convert_messages([Message.user(["read", original])])
    assert payload[-1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert original.size == (2200, 1600)


@pytest.mark.asyncio
async def test_screen_fallback_uses_vision_and_never_blind_text():
    primary = MagicMock(supports_images=True, name="gemini")
    text = MagicMock(supports_images=False, name="groq")
    backup = MagicMock(supports_images=True, name="vision")

    async def fail(*args, **kwargs):
        raise ConnectionError("503 busy")
        yield ""

    async def read(*args, **kwargs):
        yield "The screen says HELLO."

    primary.stream = fail
    backup.stream = read
    router = ModelRouter(
        {ModelTier.COMPLEX: primary, ModelTier.STANDARD: text},
        RouterSettings(),
        vision_fallback=backup,
    )
    chunks = [
        c
        async for c in router.route_stream(
            [Message.user(["read", Image.new("RGB", (10, 10))])], target_tier=ModelTier.COMPLEX
        )
    ]
    assert "The screen says HELLO." in chunks
    text.stream.assert_not_called()


@pytest.mark.asyncio
async def test_stt_connection_errors_reach_caller(tmp_path):
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"test")
    provider = STTProvider(api_key="test")
    provider.client.audio.transcriptions.create = AsyncMock(side_effect=ConnectionError("offline"))
    with pytest.raises(ConnectionError):
        await provider.transcribe(audio)


@pytest.mark.asyncio
async def test_tts_does_not_swallow_model_failure(monkeypatch):
    tts = TTSProvider()
    tts._use_pygame = False
    monkeypatch.setattr(tts, "_speech_unavailable", AsyncMock())

    async def broken():
        raise LLMError("provider unavailable")
        yield ""

    with pytest.raises(LLMError, match="unavailable"):
        await tts.speak_stream(broken())
    assert not tts.is_speaking


@pytest.mark.asyncio
async def test_builtin_tools_work_without_child_processes():
    manager = MCPManager()
    assert await manager.start_builtin("monitor", "jarvis.tools.mcp_world_monitor")
    declarations = await manager.get_gemini_tools()
    assert declarations.function_declarations
    result = await manager.call_tool("open_world_monitor", {})
    assert result and not result.startswith(("Execution error", "Error"))
    await manager.shutdown()


@pytest.mark.asyncio
async def test_websocket_survives_occupied_port_and_rejects_unauthorized_client():
    session = SimpleNamespace(
        settings=Settings(gemini_api_key="", groq_api_key=""),
        router=SimpleNamespace(providers={}),
        memory=None,
    )
    server = WebSocketServer(session, port=0)
    server._update_world_data_loop = AsyncMock()
    await server.start()
    second = WebSocketServer(session, port=server.port)
    second._update_world_data_loop = AsyncMock()
    await second.start()
    try:
        assert second.port != server.port
        async with websockets.connect(
            f"ws://127.0.0.1:{server.port}/?token={server.auth_token}"
        ) as ws:
            assert json.loads(await ws.recv())["type"] == "capabilities"
            await ws.send(json.dumps({"type": "get_memories"}))
            result = json.loads(await ws.recv())
            assert result["disabled"] is True
        async with websockets.connect(f"ws://127.0.0.1:{server.port}") as ws:
            with pytest.raises(websockets.exceptions.ConnectionClosed):
                await ws.recv()
    finally:
        await second.stop()
        await server.stop()
