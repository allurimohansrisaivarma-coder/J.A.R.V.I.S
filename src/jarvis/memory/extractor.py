"""Background memory extraction from conversations."""

from typing import TYPE_CHECKING

import structlog

from jarvis.llm.base import Message
from jarvis.memory.manager import MemoryManager

if TYPE_CHECKING:
    from jarvis.core.session import SessionManager

logger = structlog.get_logger(__name__)

EXTRACTION_PROMPT = """
You are a memory extraction sub-system. Analyze the following conversation transcript.
Your goal is to extract EVERYTHING that could possibly be useful for future context.
Extract:
- User preferences, names, habits, or facts about the user.
- Actionable tasks, plans, or ongoing projects mentioned.
- Technical context (e.g. tools used, file paths, coding patterns, errors).
- Specific details, nuances, or decisions discussed in the conversation.
- Any other detail that would help an AI assistant remember the context of this conversation.

Format each extracted detail as a self-contained sentence on a new line. 
Do not output anything else. If there is absolutely nothing to extract, output exactly "NONE".
"""

class MemoryExtractor:
    """Extracts facts from conversations in the background."""
    
    def __init__(self, session: "SessionManager", memory_manager: MemoryManager):
        """Initialize the extractor.
        
        Args:
            session: The main application session (to access LLM router).
            memory_manager: The memory manager to store facts.
        """
        self.session = session
        self.memory = memory_manager
        
    async def extract_from_messages(self, messages: list[Message]) -> None:
        """Run extraction on a list of messages. This is typically run asynchronously.
        
        Args:
            messages: A list of messages representing a conversation segment.
        """
        if not messages:
            return
            
        logger.info("Starting background memory extraction", num_messages=len(messages))
        
        # Construct transcript string
        transcript = "\n".join([f"{m.role}: {m.content}" for m in messages if m.role != "system"])
        
        prompt = [
            Message.system(EXTRACTION_PROMPT),
            Message.user(f"Transcript:\n{transcript}")
        ]
        
        try:
            # Generate extraction using the router
            response = await self.session.router.generate_with_fallback(prompt)
            content = response.content.strip()
            
            if content and content.upper() != "NONE":
                facts = [f.strip() for f in content.split("\n") if f.strip()]
                for fact in facts:
                    # Remove common markdown bullet points if present
                    if fact.startswith(("-", "*")):
                        fact = fact[1:].strip()

                    # Do not persist file dumps, prompts, or other noisy
                    # conversation artifacts as long-term memory.
                    looks_like_artifact = (
                        len(fact) > 300
                        or "```" in fact
                        or fact.startswith(("File:", "%PDF", "[System"))
                    )
                    if fact and not looks_like_artifact:
                        self.memory.store_fact(fact, source_context="Conversation transcript")
                logger.info("Extraction complete", extracted_count=len(facts))
            else:
                logger.debug("Extraction complete, nothing to extract")
                
        except Exception as e:
            logger.error("Failed to extract memory", error=str(e))
