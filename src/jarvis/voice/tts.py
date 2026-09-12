"""Text-to-speech using Microsoft Edge TTS API with streaming playback."""
# ruff: noqa: S110

import asyncio
import inspect
import os
import re
import tempfile
from collections.abc import AsyncIterator, Callable

import edge_tts
import structlog

logger = structlog.get_logger(__name__)

# pygame otherwise writes a promotional banner into the structured startup log.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

# Short, complete phrases start playback sooner than waiting for a whole model
# sentence.  This is deliberately conservative: too-small chunks sound choppy.
_SENTENCE_ENDS = {".", "!", "?", "\n"}
_CLAUSE_ENDS = {",", ";", ":"}
_MIN_PHRASE_CHARS = 28
_FIRST_PHRASE_TARGET = 72
_PHRASE_TARGET = 220
_MAX_PHRASE_OVERRUN = 40
_SYNTHESIS_ATTEMPTS = 4
_SYNTHESIS_TIMEOUT_SECONDS = 15.0
_SYNTHESIS_RETRY_DELAYS = (0.4, 1.0, 2.0)
_URL_PATTERN = r"(?:https?://|www\.)[^\s<>]+|(?<![\w@])(?:[a-z0-9-]+\.)+(?:com|org|net|edu|gov|io|co|ai|uk|in|ae)\b(?:[/?#][^\s<>]*)?"
_WINDOWS_PATH_PATTERN = r"(?<!\w)[A-Za-z]:[\\/][^\r\n`<>]+"


class TTSProvider:
    """Provides Text-to-Speech capabilities using Edge TTS with streaming support."""

    def __init__(
        self,
        voice: str = "en-GB-RyanNeural",
        rate: str = "+20%",
        pitch: str = "-5Hz",
    ):
        """Initialize the TTS provider.

        Args:
            voice: The voice ID to use (e.g., 'en-GB-RyanNeural').
            rate: Speech rate adjustment (e.g., '+0%', '-10%', '+20%').
            pitch: Pitch adjustment supported by Edge TTS (e.g., '-5Hz').
        """
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self._is_speaking = False
        self._stop_requested = False
        self._current_queue: asyncio.Queue[str | None] | None = None
        self._notice_sent = False
        self.on_unavailable: Callable[[str], object] | None = None

        # Try to initialize pygame mixer as primary playback
        self._use_pygame = False
        try:
            import pygame

            pygame.mixer.init()
            self._use_pygame = True
            logger.info("TTS initialized with pygame playback", voice=voice, rate=rate, pitch=pitch)
        except Exception as e:
            logger.warning("pygame mixer unavailable; audio playback disabled", error=str(e))

    def _clean_text(self, text: str) -> str:
        """Remove markdown artifacts that TTS would try to pronounce."""
        clean = re.sub(
            rf"`{_WINDOWS_PATH_PATTERN}`",
            "the file location shown in chat",
            text,
        )
        clean = clean.replace("*", "").replace("#", "").replace("`", "")
        clean = clean.replace("---", "").replace("___", "")
        # Keep readable source names in speech, never URL syntax or footnote IDs.
        clean = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", clean)
        clean = re.sub(_URL_PATTERN, " ", clean, flags=re.IGNORECASE)
        clean = re.sub(_WINDOWS_PATH_PATTERN, "the file location shown in chat", clean)
        clean = re.sub(r"【[^】]*】|\[\d+(?:[, -]\d+)*\]", "", clean)
        clean = re.sub(r"(?im)^\s*(?:sources?|references?|links?)\s*:\s*[(),.;\s]*$", "", clean)
        clean = re.sub(r"[<>]", "", clean)
        clean = clean.strip()
        # Edge TTS raises NoAudioReceived for formatting remnants such as a
        # standalone Markdown bullet. They are layout, not speech.
        return clean if any(character.isalnum() for character in clean) else ""

    async def _speech_unavailable(self, text: str) -> None:
        """Keep text available without silently substituting a different voice."""
        if not self._notice_sent:
            self._notice_sent = True
            message = "The selected JARVIS voice is temporarily unavailable. The reply is shown in chat; no other voice will be substituted."
            logger.warning(message)
            if self.on_unavailable:
                result = self.on_unavailable(message)
                if inspect.isawaitable(result):
                    await result

    async def _save_voice(self, text: str, path: str, voice: str, rate: str, pitch: str) -> None:
        """Retry transient synthesis failures without changing the selected voice."""
        for attempt in range(_SYNTHESIS_ATTEMPTS):
            try:
                await asyncio.wait_for(
                    edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(path),
                    timeout=_SYNTHESIS_TIMEOUT_SECONDS,
                )
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if attempt == _SYNTHESIS_ATTEMPTS - 1 or self._stop_requested:
                    raise
                logger.warning(
                    "Retrying selected voice synthesis",
                    voice=voice,
                    attempt=attempt + 2,
                    error_type=type(exc).__name__,
                )
                await asyncio.sleep(_SYNTHESIS_RETRY_DELAYS[attempt])

    async def speak(self, text: str) -> None:
        """Synthesize and play speech (blocking, full text at once).

        This is the legacy method. Use speak_stream() for real-time streaming.

        Args:
            text: The text to speak.
        """
        if not text.strip():
            return

        clean_text = self._clean_text(text)
        if not clean_text:
            return

        self._is_speaking = True
        self._stop_requested = False
        self._notice_sent = False

        if not self._use_pygame:
            try:
                await self._speech_unavailable(clean_text)
            finally:
                self._is_speaking = False
            return

        fd, temp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)

        rate_str = str(self.rate).strip()
        if not rate_str.endswith("%"):
            rate_str += "%"
        if not rate_str.startswith("+") and not rate_str.startswith("-"):
            rate_str = f"+{rate_str}"

        try:
            logger.debug("Generating TTS audio", text_length=len(clean_text), voice=self.voice)
            await self._save_voice(clean_text, temp_path, self.voice, rate_str, self.pitch)

            if self._stop_requested:
                return

            if self._use_pygame:
                import pygame

                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() and not self._stop_requested:
                    await asyncio.sleep(0.1)

        except Exception as e:
            logger.error("Error during TTS playback", error=str(e))
            await self._speech_unavailable(clean_text)
        finally:
            self._is_speaking = False
            if self._use_pygame:
                try:
                    import pygame

                    pygame.mixer.music.unload()
                except Exception:
                    pass
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                logger.debug("Could not delete temp TTS file", error=str(e))

    @staticmethod
    def _take_phrase(
        buffer: str, *, final: bool, target: int = _PHRASE_TARGET
    ) -> tuple[str | None, str]:
        """Extract the next natural TTS phrase without waiting on long sentences."""
        if not buffer.strip():
            return None, buffer

        # Do not split URLs or Markdown links between synthesis requests: a
        # fragment such as 'com/results' cannot be recognized as a URL later.
        protected = [
            m.span()
            for m in re.finditer(
                r"\[[^\]]*(?:\](?:\([^)]*(?:\)|$))?)?|"
                + _URL_PATTERN
                + "|"
                + _WINDOWS_PATH_PATTERN,
                buffer,
                re.IGNORECASE,
            )
        ]
        if not final:
            partial_domain = re.search(r"(?<![\w@])(?:[\w-]+\.)+[\w-]*(?:[/?#][^\s<>]*)?$", buffer)
            if partial_domain:
                protected.append(partial_domain.span())

        def safe_boundary(end: int) -> bool:
            return not any(
                start < end < stop or (not final and start < end == stop == len(buffer))
                for start, stop in protected
            )

        punctuation_positions = [
            index
            for index, char in enumerate(buffer)
            if char in _SENTENCE_ENDS
            and _MIN_PHRASE_CHARS <= index + 1 <= target + _MAX_PHRASE_OVERRUN
            and safe_boundary(index + 1)
        ]
        if punctuation_positions:
            end = punctuation_positions[0] + 1
            return buffer[:end].strip(), buffer[end:]

        clause_positions = [
            index
            for index, char in enumerate(buffer)
            if char in _CLAUSE_ENDS
            and _MIN_PHRASE_CHARS <= index + 1 <= target + _MAX_PHRASE_OVERRUN
            and safe_boundary(index + 1)
        ]
        if clause_positions and (len(buffer) >= target or final):
            end = clause_positions[0] + 1
            return buffer[:end].strip(), buffer[end:]

        if len(buffer) >= target:
            # Never cut through a word.  A short extension is less disruptive
            # than a clipped word, and keeps playback conversational.
            end = buffer.rfind(" ", 0, target + 1)
            while end > 0 and not safe_boundary(end):
                end = buffer.rfind(" ", 0, end)
            if end > 0:
                return buffer[:end].strip(), buffer[end:]

        if final:
            return buffer.strip(), ""
        return None, buffer

    async def speak_stream(
        self,
        text_stream: AsyncIterator[str],
        on_phrase_ready: Callable[[str], object] | None = None,
    ) -> None:
        """Consume an async text stream, buffer sentences, and speak them as they arrive.

        This method enables JARVIS to start speaking while the LLM is still generating.

        Args:
            text_stream: An async iterator yielding text chunks from the LLM.
        """
        self._is_speaking = True
        self._stop_requested = False

        self._notice_sent = False
        utterance_voice, utterance_rate, utterance_pitch = self.voice, self.rate, self.pitch
        # Pipeline queues
        text_queue: asyncio.Queue[str | None] = asyncio.Queue()
        audio_queue: asyncio.Queue[tuple[str | None, str] | None] = asyncio.Queue()
        temp_files: set[str] = set()

        # Keep track of generated temp paths so we can clean them up if stopped
        self._current_queue = text_queue

        async def _buffer_text():
            """Accumulate LLM chunks into complete phrases and enqueue them."""
            buffer = ""
            is_first_phrase = True
            try:
                async for chunk in text_stream:
                    if self._stop_requested:
                        break
                    buffer += chunk

                    while buffer:
                        phrase, buffer = self._take_phrase(
                            buffer,
                            final=False,
                            target=_FIRST_PHRASE_TARGET if is_first_phrase else _PHRASE_TARGET,
                        )
                        if not phrase:
                            break
                        await text_queue.put(phrase)
                        is_first_phrase = False

                while buffer.strip():
                    phrase, buffer = self._take_phrase(
                        buffer,
                        final=True,
                        target=_FIRST_PHRASE_TARGET if is_first_phrase else _PHRASE_TARGET,
                    )
                    if phrase:
                        await text_queue.put(phrase)
                        is_first_phrase = False
            except Exception as e:
                logger.error("Error buffering text", error=str(e))
                raise
            finally:
                await text_queue.put(None)

        async def _download_audio():
            """Pop phrases, download MP3s, and queue the file paths."""
            try:
                while not self._stop_requested:
                    phrase = await text_queue.get()
                    if phrase is None:
                        break

                    clean = self._clean_text(phrase)
                    if not clean:
                        await audio_queue.put((None, phrase))
                        continue

                    if not self._use_pygame:
                        await audio_queue.put((None, phrase))
                        continue

                    fd, temp_path = tempfile.mkstemp(suffix=".mp3")
                    os.close(fd)
                    temp_files.add(temp_path)

                    rate_str = str(utterance_rate).strip()
                    if not rate_str.endswith("%"):
                        rate_str += "%"
                    if not rate_str.startswith("+") and not rate_str.startswith("-"):
                        rate_str = f"+{rate_str}"

                    try:
                        await self._save_voice(
                            clean, temp_path, utterance_voice, rate_str, utterance_pitch
                        )

                        if self._stop_requested:
                            os.unlink(temp_path)
                            break

                        await audio_queue.put((temp_path, phrase))
                    except Exception as e:
                        logger.error("Error downloading phrase", error=str(e))
                        # Preserve phrase ordering and continue trying the same
                        # selected voice for later phrases after an isolated failure.
                        await audio_queue.put((None, phrase))
                        try:
                            os.unlink(temp_path)
                        except Exception:
                            pass
            finally:
                await audio_queue.put(None)

        async def _play_audio():
            """Pop MP3 paths and play them sequentially using pygame."""
            try:
                while not self._stop_requested:
                    audio_item = await audio_queue.get()
                    if audio_item is None:
                        break
                    temp_path, phrase = audio_item

                    if on_phrase_ready:
                        callback_result = on_phrase_ready(phrase + " ")
                        if inspect.isawaitable(callback_result):
                            await callback_result

                    if temp_path is None:
                        clean = self._clean_text(phrase)
                        if clean:
                            await self._speech_unavailable(clean)
                        continue

                    if self._use_pygame:
                        import pygame

                        try:
                            pygame.mixer.music.load(temp_path)
                            pygame.mixer.music.play()
                            while pygame.mixer.music.get_busy() and not self._stop_requested:
                                await asyncio.sleep(0.05)
                        except Exception as e:
                            logger.error("Error playing chunk", error=str(e))
                            await self._speech_unavailable(self._clean_text(phrase))
                        finally:
                            try:
                                pygame.mixer.music.unload()
                            except Exception:
                                pass
                            try:
                                if os.path.exists(temp_path):
                                    os.unlink(temp_path)
                            except Exception:
                                pass
                    elif on_phrase_ready:
                        # Keep the text channel usable even when the local
                        # audio backend is unavailable.
                        try:
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)
                        except Exception:
                            pass
            finally:
                # Cleanup any remaining files in the queue
                while not audio_queue.empty():
                    item = audio_queue.get_nowait()
                    if item:
                        path, _ = item
                        try:
                            if path and os.path.exists(path):
                                os.unlink(path)
                        except Exception:
                            pass

        try:
            buffer_task = asyncio.create_task(_buffer_text())
            download_task = asyncio.create_task(_download_audio())
            play_task = asyncio.create_task(_play_audio())

            await asyncio.gather(buffer_task, download_task, play_task)

        except Exception as e:
            logger.error("Error in streaming TTS", error=str(e))
            raise
        finally:
            self._stop_requested = True
            for task in (buffer_task, download_task, play_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(buffer_task, download_task, play_task, return_exceptions=True)
            if self._use_pygame:
                import pygame

                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            for path in temp_files:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            self._is_speaking = False
            self._current_queue = None

    def stop(self) -> None:
        """Request immediate stop of any current speech."""
        self._stop_requested = True

        if self._current_queue is not None:
            try:
                self._current_queue.put_nowait(None)
            except Exception:
                pass

        if self._use_pygame:
            try:
                import pygame

                pygame.mixer.music.stop()
            except Exception:
                pass

    @property
    def is_speaking(self) -> bool:
        """Whether the TTS is currently playing audio."""
        return self._is_speaking
