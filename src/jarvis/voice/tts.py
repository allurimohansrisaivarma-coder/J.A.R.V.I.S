"""Text-to-speech using Microsoft Edge TTS API with streaming playback."""

import asyncio
import os
import tempfile
import inspect
from collections.abc import AsyncIterator, Callable
import structlog

import edge_tts

logger = structlog.get_logger(__name__)

# Short, complete phrases start playback sooner than waiting for a whole model
# sentence.  This is deliberately conservative: too-small chunks sound choppy.
_SENTENCE_ENDS = {'.', '!', '?', '\n'}
_CLAUSE_ENDS = {',', ';', ':'}
_MIN_PHRASE_CHARS = 28
_FIRST_PHRASE_TARGET = 72
_PHRASE_TARGET = 120
_MAX_PHRASE_OVERRUN = 20


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
        
        # Try to initialize pygame mixer as primary playback
        self._use_pygame = False
        try:
            import pygame
            pygame.mixer.init(frequency=24000, channels=2)
            self._use_pygame = True
            logger.info("TTS initialized with pygame playback", voice=voice, rate=rate, pitch=pitch)
        except Exception as e:
            logger.warning("pygame mixer unavailable, will use temp file fallback", error=str(e))

    def _clean_text(self, text: str) -> str:
        """Remove markdown artifacts that TTS would try to pronounce."""
        clean = text.replace("*", "").replace("#", "").replace("`", "")
        clean = clean.replace("---", "").replace("___", "")
        # Remove markdown links: [text](url) -> text
        import re
        clean = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', clean)
        return clean.strip()

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

        fd, temp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        
        try:
            logger.debug("Generating TTS audio", text_length=len(clean_text), voice=self.voice)
            communicate = edge_tts.Communicate(
                clean_text, self.voice, rate=self.rate, pitch=self.pitch
            )
            await communicate.save(temp_path)
            
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
        finally:
            self._is_speaking = False
            if self._use_pygame:
                try:
                    import pygame
                    pygame.mixer.music.unload()
                except (AttributeError, Exception):
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

        punctuation_positions = [
            index
            for index, char in enumerate(buffer)
            if char in _SENTENCE_ENDS
            and _MIN_PHRASE_CHARS <= index + 1 <= target + _MAX_PHRASE_OVERRUN
        ]
        if punctuation_positions:
            end = punctuation_positions[0] + 1
            return buffer[:end].strip(), buffer[end:]

        clause_positions = [
            index
            for index, char in enumerate(buffer)
            if char in _CLAUSE_ENDS
            and _MIN_PHRASE_CHARS <= index + 1 <= target + _MAX_PHRASE_OVERRUN
        ]
        if clause_positions and (len(buffer) >= target or final):
            end = clause_positions[0] + 1
            return buffer[:end].strip(), buffer[end:]

        if len(buffer) >= target:
            # Never cut through a word.  A short extension is less disruptive
            # than a clipped word, and keeps playback conversational.
            end = buffer.rfind(" ", 0, target + 1)
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
        
        # Pipeline queues
        text_queue: asyncio.Queue[str | None] = asyncio.Queue()
        audio_queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
        
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
                        continue
                        
                    fd, temp_path = tempfile.mkstemp(suffix=".mp3")
                    os.close(fd)
                    
                    try:
                        communicate = edge_tts.Communicate(
                            clean, self.voice, rate=self.rate, pitch=self.pitch
                        )
                        await communicate.save(temp_path)
                        
                        if self._stop_requested:
                            os.unlink(temp_path)
                            break
                            
                        await audio_queue.put((temp_path, phrase))
                    except Exception as e:
                        logger.error("Error downloading phrase", error=str(e))
                        # Do not leave the chat stream blank if synthesis is
                        # temporarily unavailable; the text is still useful.
                        if on_phrase_ready:
                            callback_result = on_phrase_ready(phrase)
                            if inspect.isawaitable(callback_result):
                                await callback_result
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
                        
                    if self._use_pygame:
                        import pygame
                        try:
                            pygame.mixer.music.load(temp_path)
                            if on_phrase_ready:
                                callback_result = on_phrase_ready(phrase)
                                if inspect.isawaitable(callback_result):
                                    await callback_result
                            pygame.mixer.music.play()
                            while pygame.mixer.music.get_busy() and not self._stop_requested:
                                await asyncio.sleep(0.05)
                        except Exception as e:
                            logger.error("Error playing chunk", error=str(e))
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
                        callback_result = on_phrase_ready(phrase)
                        if inspect.isawaitable(callback_result):
                            await callback_result
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
                            if os.path.exists(path):
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
        finally:
            self._is_speaking = False
            self._current_queue = None
    
    async def _speak_sentence(self, sentence: str) -> None:
        """Synthesize and play a single sentence.
        
        Uses temp file + pygame approach (reliable on Windows).
        
        Args:
            sentence: Clean text sentence to speak.
        """
        if self._stop_requested or not sentence:
            return
            
        fd, temp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        
        try:
            communicate = edge_tts.Communicate(
                sentence, self.voice, rate=self.rate, pitch=self.pitch
            )
            await communicate.save(temp_path)
            
            if self._stop_requested:
                return
            
            if self._use_pygame:
                import pygame
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() and not self._stop_requested:
                    await asyncio.sleep(0.05)
                    
        except Exception as e:
            logger.error("Error speaking sentence", error=str(e), sentence_len=len(sentence))
        finally:
            if self._use_pygame:
                try:
                    import pygame
                    pygame.mixer.music.unload()
                except (AttributeError, Exception):
                    pass
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                pass

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
