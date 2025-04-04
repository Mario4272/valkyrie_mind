from abc import ABC, abstractmethod
from typing import Any
from .context_types import ContextData

class BaseContextProcessor(ABC):
    """Abstract base class for all context processors."""

    @abstractmethod
    def process(self, data: Any) -> ContextData:
        """Process incoming data and return a ContextData object."""
        pass