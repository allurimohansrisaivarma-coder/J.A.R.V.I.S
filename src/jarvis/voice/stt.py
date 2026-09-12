"""Speech-to-text using Groq Whisper API."""

import asyncio
from pathlib import Path

import structlog
from groq import AsyncGroq

logger = structlog.get_logger(__name__)


class STTProvider:
    """Provides Speech-to-Text capabilities using Groq's fast Whisper API."""

    def __init__(
        self,
        api_key: str,
        model: str = "whisper-large-v3-turbo",
        language: str | None = "en",
    ):
        """Initialize STT provider.

        Args:
            api_key: Groq API key
        """
        self.client = AsyncGroq(api_key=api_key, timeout=15.0, max_retries=0)
        self.model = model
        self.language = language

    async def transcribe(self, audio_file_path: Path) -> str:
        """Transcribe an audio file to text.

        Args:
            audio_file_path: Path to the audio file (e.g. wav format).

        Returns:
            The transcribed text. Service failures propagate to the UI.
        """
        logger.debug("Transcribing audio", path=str(audio_file_path))

        try:
            audio_bytes = await asyncio.to_thread(audio_file_path.read_bytes)
            request: dict[str, object] = {
                "file": (audio_file_path.name, audio_bytes),
                "model": self.model,
            }
            if self.language:
                request["language"] = self.language
            transcription = await asyncio.wait_for(
                self.client.audio.transcriptions.create(**request), timeout=18.0
            )

            result = transcription.text.strip()
            logger.info("Transcription complete", length=len(result))
            return result
        except Exception as e:
            logger.error("STT transcription failed", error=str(e))
            raise
