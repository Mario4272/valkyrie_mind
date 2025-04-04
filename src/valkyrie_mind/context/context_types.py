from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional
import numpy as np
from abc import ABC, abstractmethod

class ContextType(Enum):
    ENVIRONMENTAL = "environmental"  # Physical environment
    SITUATIONAL = "situational"      # Current situation
    TEMPORAL = "temporal"            # Time-related
    SOCIAL = "social"                # Social interactions
    TASK = "task"                    # Current tasks
    HISTORICAL = "historical"        # Past contexts
    EMOTIONAL = "emotional"          # Emotional state
    COGNITIVE = "cognitive"          # Mental state

@dataclass
class ContextData:
    """Represents a specific context"""
    context_type: ContextType
    timestamp: datetime
    data: dict
    confidence: float
    sources: List[str]
    relevance_score: float
    duration: Optional[float] = None
    related_contexts: List[str] = None
    priority: float = 1.0
