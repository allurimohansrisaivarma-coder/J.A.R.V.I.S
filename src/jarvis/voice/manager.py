"""Voice conversation state machine and manager."""

import asyncio
import time
from collections.abc import Callable
from contextlib import suppress
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import structlog

from jarvis.utils.errors import user_error
from jarvis.utils.metrics import metrics
from jarvis.voice.capture import AudioCapture
from jarvis.voice.stt import STTProvider
from jarvis.voice.tts import TTSProvider

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from jarvis.core.session import SessionManager


class VoiceState(Enum):
    """States of the voice conversation."""

    IDLE = auto()
    LISTENING = auto()
    RECORDING = auto()
    THINKING = auto()
    SPEAKING = auto()


class VoiceManager:
    """Manages the full voice interaction loop with streaming support."""

    def __init__(self, session: "SessionManager", stt: STTProvider, tts: TTSProvider):
        """Initialize the voice manager.

        Args:
            session: Active LLM session manager.
            stt: Speech-to-Text provider.
            tts: Text-to-Speech provider.
        """
        self.session = session
        self.stt = stt
        self.tts = tts
        self.capture: AudioCapture | None = None
        self.state = VoiceState.IDLE
        self._is_running = False
        self._task: asyncio.Task | None = None
        self._active_speak_task: asyncio.Task | None = None
        self._response_lock = asyncio.Lock()

        # Push-to-talk queue (filled by hotkey listener)
        self.hotkey_queue: asyncio.Queue[str] = asyncio.Queue()

        # Callbacks for UI updates
        self.on_state_change: Callable[[VoiceState], object] | None = None
        self.on_user_message: Callable[[str], object] | None = None
        self.on_jarvis_chunk: Callable[[Any], object] | None = None
        self.on_jarvis_done: Callable[[], object] | None = None
        self.on_audio_level: Callable[[float], object] | None = None
        self.on_error: Callable[[str], object] | None = None
        self.tts.on_unavailable = self._notify_error

    def _notify_error(self, message: str) -> None:
        if self.on_error:
            self.on_error(message)

    def _set_state(self, new_state: VoiceState):
        """Update state and notify observers."""
        if self.state != new_state:
            logger.debug("Voice state changed", old_state=self.state.name, new_state=new_state.name)
            self.state = new_state
            if self.on_state_change:
                try:
                    self.on_state_change(self.state)
                except Exception as e:
                    logger.error("Error in state change callback", error=str(e))

    def interrupt_speech(self) -> bool:
        """Stop the current reply so a push-to-talk turn can take priority."""
        is_active = self.tts.is_speaking or (
            self._active_speak_task is not None and not self._active_speak_task.done()
        )
        if not is_active:
            return False

        logger.info("Push-to-talk interrupted JARVIS playback")
        self.tts.stop()
        if self._active_speak_task and not self._active_speak_task.done():
            self._active_speak_task.cancel()
        return True

    def _notify_user_message(self, text: str) -> None:
        if self.on_user_message:
            try:
                self.on_user_message(text)
            except Exception as e:
                logger.error("Error in on_user_message callback", error=str(e))

    def _notify_jarvis_phrase(self, phrase: str) -> None:
        """Publish text when its matching audio is about to be transmitted."""
        self._set_state(VoiceState.SPEAKING)
        if not getattr(self, "_ttfa_recorded", False) and hasattr(metrics, "_request_start_time"):
            metrics.record_ttfa(time.perf_counter() - metrics._request_start_time)
            self._ttfa_recorded = True

        if self.on_jarvis_chunk:
            try:
                self.on_jarvis_chunk(phrase)
            except Exception as e:
                logger.error("Error in on_jarvis_chunk callback", error=str(e))

    async def respond_to_text(self, text: str) -> str:
        """Stream an answer into speech and the UI as one live transmission.

        The UI receives each phrase immediately before its audio starts, rather
        than receiving the entire model stream ahead of the voice.
        """
        import time

        async with self._response_lock:
            self._active_speak_task = asyncio.current_task()
            self._set_state(VoiceState.THINKING)
            self._ttfa_recorded = False
            started_at = time.perf_counter()
            response_chunks: list[str] = []

            async def text_stream():
                async for chunk in self.session.process_input_stream(text):
                    if isinstance(chunk, dict):
                        if self.on_jarvis_chunk:
                            # Bypass TTS and send straight to UI
                            self.on_jarvis_chunk(chunk)
                    else:
                        response_chunks.append(chunk)
                        yield chunk

            try:
                await self.tts.speak_stream(text_stream(), self._notify_jarvis_phrase)
            except asyncio.CancelledError:
                logger.info("Voice response interrupted")
                raise
            finally:
                response = "".join(response_chunks)
                self._set_state(VoiceState.IDLE)
                if self.on_jarvis_done:
                    try:
                        self.on_jarvis_done()
                    except Exception as e:
                        logger.error("Error in on_jarvis_done callback", error=str(e))
                logger.info(
                    "Jarvis voice response completed",
                    response_len=len(response),
                    latency=time.perf_counter() - started_at,
                )
                metrics.end_request()
            return "".join(response_chunks)

    async def start_ptt_loop(self):
        """Push-to-talk mode: record only while hotkey is held."""
        if getattr(self, "_starting_ptt", False) or self._is_running:
            return

        self._starting_ptt = True
        try:
            self._is_running = True
            self._task = asyncio.current_task()
            self._set_state(VoiceState.IDLE)
        finally:
            self._starting_ptt = False
        logger.info("Starting push-to-talk loop")

        try:
            while self._is_running:
                # Wait for PTT_START event from hotkey listener
                event = await self.hotkey_queue.get()

                if event == "PTT_START":
                    self._set_state(VoiceState.RECORDING)
                    logger.info("PTT recording started")

                    # Record until PTT_STOP
                    if self.capture is None:
                        self.capture = AudioCapture()
                    audio_path = await self.capture.record_until_released(
                        self.hotkey_queue, on_audio_level=self.on_audio_level
                    )

                    if not audio_path or not self._is_running:
                        if self._is_running:
                            self._notify_error(
                                "No microphone audio captured. Check the input device and microphone permissions."
                            )
                        self._set_state(VoiceState.IDLE)
                        continue

                    # STT
                    self._set_state(VoiceState.THINKING)
                    metrics.start_request("voice")
                    stt_start = time.perf_counter()
                    try:
                        text = await self.stt.transcribe(audio_path)
                    except Exception as exc:
                        self._notify_error(user_error(exc, operation="speech recognition"))
                        metrics.end_request(status="stt_failed")
                        self._set_state(VoiceState.IDLE)
                        continue
                    finally:
                        with suppress(OSError):
                            audio_path.unlink()
                    metrics.record_stage("stt", time.perf_counter() - stt_start)

                    try:
                        audio_path.unlink()
                    except OSError as exc:
                        logger.debug("Could not delete captured audio", error=str(exc))

                    if not text:
                        self._notify_error(
                            "No speech detected. Hold the mic button while speaking, then release it."
                        )
                        metrics.end_request(status="empty_transcript")
                        self._set_state(VoiceState.IDLE)
                        continue

                    logger.info("User said (PTT)", text=text)
                    self._notify_user_message(text)

                    # Cancel any existing speak task just in case
                    if self._active_speak_task and not self._active_speak_task.done():
                        self.tts.stop()
                        self._active_speak_task.cancel()

                    self._active_speak_task = asyncio.create_task(self.respond_to_text(text))

                    # Wait for either the speaking to finish, or the next hotkey event
                    speak_task = self._active_speak_task
                    hotkey_task = asyncio.create_task(self.hotkey_queue.get())

                    done, _pending = await asyncio.wait(
                        [speak_task, hotkey_task], return_when=asyncio.FIRST_COMPLETED
                    )

                    if hotkey_task in done:
                        # User interrupted JARVIS!
                        event = hotkey_task.result()
                        if event == "PTT_START":
                            logger.info("JARVIS interrupted by user!")
                            self.tts.stop()
                            if not speak_task.done():
                                speak_task.cancel()

                            # Put the event back so the next loop iteration catches it
                            self.hotkey_queue.put_nowait("PTT_START")
                    else:
                        # JARVIS finished speaking normally
                        hotkey_task.cancel()

                    for pending_task in (speak_task, hotkey_task):
                        if not pending_task.done():
                            pending_task.cancel()
                        with suppress(asyncio.CancelledError):
                            try:
                                await pending_task
                            except Exception as exc:
                                self._notify_error(user_error(exc))

                    if self.state == VoiceState.SPEAKING:
                        self._set_state(VoiceState.IDLE)

        except asyncio.CancelledError:
            logger.info("PTT loop cancelled")
        except Exception as e:
            logger.error("Error in PTT loop", error=str(e))
            self._notify_error(user_error(e, operation="voice input"))
        finally:
            if self._task is asyncio.current_task():
                self._is_running = False
                self._set_state(VoiceState.IDLE)

    def stop(self):
        """Stop the voice loop."""
        self._is_running = False
        self.tts.stop()
        if self._active_speak_task and not self._active_speak_task.done():
            self._active_speak_task.cancel()
        if self._task and not self._task.done():
            self._task.cancel()

    async def cancel_turn(self) -> None:
        """Cancel capture/transcription/playback and re-arm the global hotkey."""
        task = self._task
        response_task = self._active_speak_task
        self.stop()
        pending = [t for t in (task, response_task) if t and t is not asyncio.current_task()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        while not self.hotkey_queue.empty():
            self.hotkey_queue.get_nowait()
        self._set_state(VoiceState.IDLE)
        asyncio.create_task(self.start_ptt_loop())
