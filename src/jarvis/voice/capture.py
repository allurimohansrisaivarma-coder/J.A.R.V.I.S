"""Audio capture and Voice Activity Detection (VAD)."""

import asyncio
import collections
import inspect
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sounddevice as sd
import soundfile as sf
import structlog

logger = structlog.get_logger(__name__)

VAD_MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
VAD_MODEL_PATH = Path(__file__).parent / "silero_vad.onnx"


class SileroVAD:
    """Wrapper around the Silero VAD ONNX model."""

    def __init__(self):
        """Initialize the VAD model. Downloads it if missing."""
        self._ensure_model_exists()

        # Suppress onnxruntime warnings
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3

        self.session = ort.InferenceSession(str(VAD_MODEL_PATH), sess_options)
        self.reset_states()

    def _ensure_model_exists(self):
        """Download the model if it doesn't exist locally."""
        if not VAD_MODEL_PATH.exists():
            logger.info("Downloading Silero VAD model", url=VAD_MODEL_URL)
            urllib.request.urlretrieve(VAD_MODEL_URL, VAD_MODEL_PATH)
            logger.info("Silero VAD model downloaded")

    def reset_states(self):
        """Reset the internal states of the RNN."""
        self._state: np.ndarray = np.zeros((2, 1, 128)).astype("float32")

    def is_speech(self, audio_chunk: np.ndarray, sr: int = 16000) -> float:
        """Calculate the probability that the chunk contains speech.

        Args:
            audio_chunk: A 1D float32 numpy array normalized between -1 and 1.
            sr: Sample rate (must be 16000 for standard Silero VAD).

        Returns:
            Probability of speech between 0.0 and 1.0.
        """
        ort_inputs = {
            "input": audio_chunk.reshape(1, -1),
            "sr": np.array(sr, dtype="int64"),
            "state": self._state,
        }
        ort_outs = self.session.run(None, ort_inputs)
        out, self._state = ort_outs
        return float(out[0][0])


class AudioCapture:
    """Captures audio from the microphone and yields spoken segments."""

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 512):
        """Initialize audio capture.

        Args:
            sample_rate: Required 16000Hz for Silero VAD.
            chunk_size: Processing chunk size. 512 is 32ms at 16kHz.
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.vad = SileroVAD()

    async def listen_for_speech(
        self, silence_duration: float = 1.5, on_recording_start=None, on_audio_level=None
    ) -> Path | None:
        """Listen to the microphone and record a single utterance.

        Args:
            silence_duration: How many seconds of silence indicates the end of speech.
            on_recording_start: Optional callback when speech is first detected.

        Returns:
            Path to the saved WAV file containing the speech, or None if cancelled.
        """
        self.vad.reset_states()

        # Audio buffer for the current utterance
        audio_buffer: list[np.ndarray] = []
        is_recording = False
        speech_streak = 0
        max_silence_chunks = int((silence_duration * self.sample_rate) / self.chunk_size)

        # We need an asyncio Queue to bridge the callback-based sounddevice with async
        q: asyncio.Queue[np.ndarray] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        # Rolling window to track speech over the silence duration
        speech_window: collections.deque[bool] = collections.deque(maxlen=max_silence_chunks)

        # Fallback buffer: keep up to 30 seconds of audio just in case VAD fails entirely
        ptt_buffer: collections.deque[np.ndarray] = collections.deque(
            maxlen=int(30 * self.sample_rate / self.chunk_size)
        )

        def audio_callback(indata: np.ndarray, frames: int, time, status: sd.CallbackFlags):
            """Called by sounddevice for each chunk of audio."""
            if status:
                logger.warning("Audio capture status", status=str(status))

            mono_data = indata[:, 0].copy()
            loop.call_soon_threadsafe(q.put_nowait, mono_data)

            if on_audio_level:
                # Calculate max amplitude (0.0 to 1.0)
                vol = float(np.max(np.abs(mono_data)))

                def notify_level() -> None:
                    try:
                        result = on_audio_level(vol)
                        if inspect.isawaitable(result):
                            asyncio.ensure_future(result)
                    except Exception as exc:
                        logger.warning("Audio level callback failed", error=str(exc))

                loop.call_soon_threadsafe(notify_level)

        logger.info("Starting audio capture", device=sd.default.device)

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_size,
                callback=audio_callback,
            ):
                logger.debug("Listening for speech...")

                # Clear keyboard buffer
                if sys.platform == "win32":
                    import msvcrt

                    while msvcrt.kbhit():
                        msvcrt.getch()

                while True:
                    # Manual cutoff check (Use Enter instead of Space because IDE terminals buffer stdin)
                    if sys.platform == "win32":
                        if msvcrt.kbhit():
                            key = msvcrt.getch()
                            if key in (b"\r", b"\n"):
                                logger.debug("Manual cutoff triggered via keyboard")
                                if not audio_buffer:
                                    logger.debug("VAD failed to trigger, using fallback PTT buffer")
                                    audio_buffer = list(ptt_buffer)
                                break

                    # Get the next chunk from the queue
                    try:
                        chunk = await asyncio.wait_for(q.get(), timeout=0.05)
                    except TimeoutError:
                        continue

                    ptt_buffer.append(chunk)

                    vad_chunk = np.clip(chunk * 20.0, -1.0, 1.0)
                    speech_prob = self.vad.is_speech(vad_chunk, self.sample_rate)
                    is_speech = speech_prob > 0.5
                    speech_window.append(is_speech)

                    if is_speech:
                        speech_streak += 1
                    else:
                        speech_streak = 0

                    if speech_streak >= 3 and not is_recording:
                        logger.debug("Speech detected, starting recording")
                        is_recording = True
                        if on_recording_start:
                            try:
                                result = on_recording_start()
                                if inspect.isawaitable(result):
                                    await result
                            except Exception as exc:
                                logger.warning("Recording callback failed", error=str(exc))

                    if is_recording:
                        audio_buffer.append(chunk)

                        # Check if the rolling window is full and mostly silent
                        if len(speech_window) == speech_window.maxlen:
                            speech_ratio = sum(speech_window) / len(speech_window)
                            # If less than 10% of the last 1.5 seconds contained speech, we are done
                            if speech_ratio < 0.1:
                                logger.debug("Silence detected, stopping recording")
                                break

        except asyncio.CancelledError:
            logger.info("Audio capture cancelled")
            return None
        except Exception as e:
            logger.error("Error during audio capture", error=str(e))
            return None

        if not audio_buffer:
            return None

        # Concatenate and save
        full_audio = np.concatenate(audio_buffer)

        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        try:
            sf.write(temp_path, full_audio, self.sample_rate)
            logger.debug(
                "Saved audio utterance", path=temp_path, duration=len(full_audio) / self.sample_rate
            )
            return Path(temp_path)
        except Exception as e:
            logger.error("Failed to save audio", error=str(e))
            return None

    async def record_until_released(
        self, event_queue: asyncio.Queue, on_audio_level=None
    ) -> Path | None:
        """Record audio until a PTT_STOP event is received.

        Unlike listen_for_speech(), this method does not use VAD.
        It simply records everything until the hotkey is released.

        Args:
            event_queue: Queue that will receive "PTT_STOP" when recording should end.

        Returns:
            Path to saved WAV file, or None on error.
        """
        audio_buffer: list[np.ndarray] = []
        q: asyncio.Queue[np.ndarray] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def audio_callback(indata: np.ndarray, frames: int, time, status: sd.CallbackFlags):
            if status:
                logger.warning("Audio capture status", status=str(status))
            mono_data = indata[:, 0].copy()
            loop.call_soon_threadsafe(q.put_nowait, mono_data)
            if on_audio_level:
                volume = float(np.max(np.abs(mono_data)))

                def notify_level() -> None:
                    try:
                        result = on_audio_level(volume)
                        if inspect.isawaitable(result):
                            asyncio.ensure_future(result)
                    except Exception as exc:
                        logger.warning("Audio level callback failed", error=str(exc))

                loop.call_soon_threadsafe(notify_level)

        logger.info("PTT recording started")

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_size,
                callback=audio_callback,
            ):
                while True:
                    # Check for stop event (non-blocking)
                    try:
                        event = event_queue.get_nowait()
                        if event == "PTT_STOP":
                            logger.debug("PTT stop received")
                            break
                    except asyncio.QueueEmpty:
                        pass

                    # Grab audio chunk
                    try:
                        chunk = await asyncio.wait_for(q.get(), timeout=0.05)
                        audio_buffer.append(chunk)
                    except TimeoutError:
                        continue

        except asyncio.CancelledError:
            logger.info("PTT recording cancelled")
            return None
        except Exception as e:
            logger.error("Error during PTT recording", error=str(e))
            return None

        if not audio_buffer:
            return None

        full_audio = np.concatenate(audio_buffer)

        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        try:
            sf.write(temp_path, full_audio, self.sample_rate)
            logger.debug(
                "Saved PTT audio", path=temp_path, duration=len(full_audio) / self.sample_rate
            )
            return Path(temp_path)
        except Exception as e:
            logger.error("Failed to save PTT audio", error=str(e))
            return None
