"""Speech-to-text using Groq Whisper API."""

from pathlib import Path
import structlog
from groq import AsyncGroq

logger = structlog.get_logger(__name__)

class STTProvider:
    """Provides Speech-to-Text capabilities using Groq's fast Whisper API."""
    
    def __init__(self, api_key: str):
        """Initialize STT provider.
        
        Args:
            api_key: Groq API key
        """
        self.client = AsyncGroq(api_key=api_key)
        self.model = "whisper-large-v3"
        
    async def transcribe(self, audio_file_path: Path) -> str:
        """Transcribe an audio file to text.
        
        Args:
            audio_file_path: Path to the audio file (e.g. wav format).
            
        Returns:
            The transcribed text, or empty string on error.
        """
        logger.debug("Transcribing audio", path=str(audio_file_path))
        
        try:
            with open(audio_file_path, "rb") as file:
                transcription = await self.client.audio.transcriptions.create(
                    file=(audio_file_path.name, file.read()),
                    model=self.model,
                    language="en",  # Enforce English for maximum speed
                )
                
            result = transcription.text.strip()
            logger.info("Transcription complete", length=len(result), text=result)
            return result
        except Exception as e:
            logger.error("STT transcription failed", error=str(e))
            return ""
