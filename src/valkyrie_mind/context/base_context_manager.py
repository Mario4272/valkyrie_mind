from abc import ABC, abstractmethod
from typing import List, Optional
from .context_types import ContextData, ContextType

class BaseContextManager(ABC):
    """Abstract base class for managing context data."""

    @abstractmethod
    def update_context(self, data: ContextData) -> None:
        """Update or insert new context data."""
        pass

    @abstractmethod
    def get_context_by_type(self, context_type: ContextType) -> List[ContextData]:
        """Retrieve context data by type."""
        pass

    @abstractmethod
    def remove_stale_context(self, threshold_seconds: float) -> None:
        """Remove context entries older than a certain time threshold."""
        pass

    @abstractmethod
    def summarize_context(self) -> str:
        """Return a string summary of current context state."""
        pass