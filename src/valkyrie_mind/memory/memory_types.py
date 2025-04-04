from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime


class MemoryType(Enum):
    SENSORY = "sensory"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    PROCEDURAL = "procedural"
    EMOTIONAL = "emotional"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


@dataclass
class Memory:
    """Represents a memory entry across various types."""
    content: Any
    memory_type: MemoryType
    timestamp: datetime
    confidence: float
    emotional_valence: Dict[str, float]
    associations: List[str]
    source_systems: List[str]
    retrieval_count: int = 0
    last_accessed: Optional[datetime] = None