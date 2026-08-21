"""Tests for the voice interface modules."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis.core.session import SessionManager
from jarvis.voice.manager import VoiceManager, VoiceState
from jarvis.voice.stt import STTProvider
from jarvis.voice.tts import TTSProvider


@pytest.fixture
def mock_session():
    session = MagicMock(spec=SessionManager)
    session.process_input = AsyncMock(return_value="Hello there.")
    return session


@pytest.fixture
def mock_stt():
    stt = MagicMock(spec=STTProvider)
    stt.transcribe = AsyncMock(return_value="Hi Jarvis")
    return stt


@pytest.fixture
def mock_tts():
    tts = MagicMock(spec=TTSProvider)
    tts.speak = AsyncMock()
    return tts


@pytest.fixture
def voice_manager(mock_session, mock_stt, mock_tts):
    manager = VoiceManager(mock_session, mock_stt, mock_tts)
    # Mock capture to prevent actual microphone usage
    manager.capture = MagicMock()

    async def wait_for_audio(*args, **kwargs):
        await asyncio.Event().wait()

    manager.capture.listen_for_speech = AsyncMock(side_effect=wait_for_audio)
    return manager


def test_initial_state(voice_manager):
    """Test the initial state is IDLE."""
    assert voice_manager.state == VoiceState.IDLE


@pytest.mark.asyncio
async def test_ptt_loop_cancellation(voice_manager):
    """Test that the push-to-talk loop exits gracefully when cancelled."""
    # Start loop in a task
    import asyncio

    task = asyncio.create_task(voice_manager.start_ptt_loop())

    # Wait a tiny bit for it to enter the loop
    await asyncio.sleep(0.01)
    assert voice_manager.state == VoiceState.IDLE

    # Stop it
    voice_manager.stop()
    await task

    # Should revert to IDLE
    assert voice_manager.state == VoiceState.IDLE


def test_phrase_buffer_starts_before_a_long_sentence():
    """The first spoken phrase should not wait for a long sentence to end."""
    text = (
        "This is a deliberately long response without terminal punctuation yet "
        + "and it keeps going " * 12
        + "."
    )
    phrase, remainder = TTSProvider._take_phrase(text, final=False, target=72)

    assert phrase is not None
    assert len(phrase) <= 72
    assert not phrase.endswith(".")
    assert remainder


@pytest.mark.asyncio
async def test_voice_response_publishes_audio_ready_phrase(mock_session, mock_stt, mock_tts):
    """Typed turns use the same synchronized text-and-speech path as voice turns."""
    manager = VoiceManager(mock_session, mock_stt, mock_tts)
    announced = []

    async def model_stream():
        yield "Good morning, "
        yield "Sir."

    async def streaming_tts(text_stream, on_phrase_ready):
        async for _ in text_stream:
            pass
        on_phrase_ready("Good morning, Sir.")

    mock_session.process_input_stream = lambda _text: model_stream()
    mock_tts.speak_stream = AsyncMock(side_effect=streaming_tts)
    manager.on_jarvis_chunk = announced.append

    response = await manager.respond_to_text("Hello")

    assert response == "Good morning, Sir."
    assert announced == ["Good morning, Sir."]


@pytest.mark.asyncio
async def test_interrupt_stops_active_speech(mock_session, mock_stt, mock_tts):
    """A new held PTT turn cancels JARVIS before recording begins."""
    manager = VoiceManager(mock_session, mock_stt, mock_tts)
    mock_tts.is_speaking = True
    manager._active_speak_task = asyncio.create_task(asyncio.Event().wait())

    assert manager.interrupt_speech() is True
    assert manager._active_speak_task.cancelled() is False
    with pytest.raises(asyncio.CancelledError):
        await manager._active_speak_task
    mock_tts.stop.assert_called_once()


def test_state_callbacks(voice_manager):
    """Test that state callbacks are triggered properly."""
    callbacks = []

    def callback(state):
        callbacks.append(state)

    voice_manager.on_state_change = callback

    voice_manager._set_state(VoiceState.LISTENING)
    assert callbacks == [VoiceState.LISTENING]

    voice_manager._set_state(VoiceState.THINKING)
    assert callbacks == [VoiceState.LISTENING, VoiceState.THINKING]
