"""Base interface for context sources."""

from abc import ABC, abstractmethod
from typing import Any

class ContextSource(ABC):
    """Interface that all context providers must implement."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the context source."""
        pass
        
    @abstractmethod
    async def can_handle(self, query: str) -> bool:
        """Determine if this source can provide relevant context for the query.
        
        Args:
            query: The user's input.
            
        Returns:
            True if this source should be invoked, False otherwise.
        """
        pass
        
    @abstractmethod
    async def gather_context(self, query: str) -> str:
        """Gather and format context for the given query.
        
        Args:
            query: The user's input.
            
        Returns:
            A string containing the gathered context, formatted for injection into the prompt.
            Return an empty string if nothing useful was found.
        """
        pass
