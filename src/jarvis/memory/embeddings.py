"""Embeddings generation using local Sentence Transformers."""

import structlog
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger(__name__)

class Embedder:
    """Generates vector embeddings for text locally."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", api_keys=None):
        """Initialize the local embedder.
        
        Args:
            model_name: The SentenceTransformer model to use.
            api_keys: Maintained for backwards compatibility, ignored.
        """
        self.model_name = model_name
        logger.info("Loading local embedding model", model=self.model_name)
        # This will download weights to ~/.cache/huggingface on first run
        self.model = SentenceTransformer(self.model_name)
        
    def embed_text(self, text: str) -> list[float]:
        """Generate a vector embedding for a single string of text to be stored."""
        try:
            vector = self.model.encode(text)
            return vector.tolist()
        except Exception as e:
            logger.error("Failed to generate local embedding", error=str(e))
            return []
            
    def embed_query(self, text: str) -> list[float]:
        """Generate a vector embedding for a search query."""
        try:
            vector = self.model.encode(text)
            return vector.tolist()
        except Exception as e:
            logger.error("Failed to generate local query embedding", error=str(e))
            return []
