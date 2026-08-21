"""Base interface for context sources."""

from abc import ABC, abstractmethod


class ContextSource(ABC):
    """Interface that all context providers must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the context source."""

    @abstractmethod
    async def can_handle(self, query: str) -> bool:
        """Determine if this source can provide relevant context for the query.

        Args:
            query: The user's input.

        Returns:
            True if this source should be invoked, False otherwise.
        """

    @abstractmethod
    async def gather_context(self, query: str, **kwargs) -> str | tuple[str, list]:
        """Gather and format context for the given query.

        Args:
            query: The user's input.
            **kwargs: Additional context variables (e.g., router for LLM expansion).

        Returns:
            A string containing the gathered context, or a tuple of (context_string, image_list).
            Return an empty string if nothing useful was found.
        """
