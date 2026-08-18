"""Context Engine module.

Gathers dynamic information from memory, web, and files to form Jarvis's World View.
"""

from jarvis.context.base import ContextSource
from jarvis.context.engine import ContextEngine

__all__ = ["ContextEngine", "ContextSource"]
