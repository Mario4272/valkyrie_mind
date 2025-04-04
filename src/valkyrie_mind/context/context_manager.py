from typing import Dict, List
from datetime import datetime, timedelta

from .context_types import ContextData, ContextType
from .base_context_manager import BaseContextManager
from .base_context_processor import BaseContextProcessor

# These would be real imports from the processors/ subpackage eventually
# For now, we mock them as simple passthroughs
class EnvironmentalContextProcessor(BaseContextProcessor):
    def process(self, data) -> ContextData:
        return ContextData(
            context_type=ContextType.ENVIRONMENTAL,
            timestamp=datetime.now(),
            data=data,
            confidence=0.9,
            sources=["mock"],
            relevance_score=1.0
        )

class SituationalContextProcessor(BaseContextProcessor):
    def process(self, data) -> ContextData:
        return ContextData(
            context_type=ContextType.SITUATIONAL,
            timestamp=datetime.now(),
            data=data,
            confidence=0.85,
            sources=["mock"],
            relevance_score=0.95
        )

class ContextManagementSystem(BaseContextManager):
    """Concrete implementation of the context manager."""
    def __init__(self):
        self.context_store: Dict[ContextType, List[ContextData]] = {
            ctype: [] for ctype in ContextType
        }
        self.environmental_processor = EnvironmentalContextProcessor()
        self.situational_processor = SituationalContextProcessor()
        # Add additional processors here

    def update_context(self, data: ContextData) -> None:
        self.context_store[data.context_type].append(data)

    def get_context_by_type(self, context_type: ContextType) -> List[ContextData]:
        return self.context_store.get(context_type, [])

    def remove_stale_context(self, threshold_seconds: float) -> None:
        cutoff = datetime.now() - timedelta(seconds=threshold_seconds)
        for ctype, items in self.context_store.items():
            self.context_store[ctype] = [item for item in items if item.timestamp >= cutoff]

    def summarize_context(self) -> str:
        summary = []
        for ctype, items in self.context_store.items():
            summary.append(f"{ctype.name}: {len(items)} entries")
        return "\n".join(summary)


# Additional context processors

class TemporalContextProcessor(BaseContextProcessor):
    def process(self, data) -> ContextData:
        return ContextData(
            context_type=ContextType.TEMPORAL,
            timestamp=datetime.now(),
            data=data,
            confidence=0.88,
            sources=["mock"],
            relevance_score=1.0
        )

class SocialContextProcessor(BaseContextProcessor):
    def process(self, data) -> ContextData:
        return ContextData(
            context_type=ContextType.SOCIAL,
            timestamp=datetime.now(),
            data=data,
            confidence=0.87,
            sources=["mock"],
            relevance_score=1.0
        )

class TaskContextProcessor(BaseContextProcessor):
    def process(self, data) -> ContextData:
        return ContextData(
            context_type=ContextType.TASK,
            timestamp=datetime.now(),
            data=data,
            confidence=0.92,
            sources=["mock"],
            relevance_score=1.0
        )

class HistoricalContextProcessor(BaseContextProcessor):
    def process(self, data) -> ContextData:
        return ContextData(
            context_type=ContextType.HISTORICAL,
            timestamp=datetime.now(),
            data=data,
            confidence=0.8,
            sources=["mock"],
            relevance_score=1.0
        )

class EmotionalContextProcessor(BaseContextProcessor):
    def process(self, data) -> ContextData:
        return ContextData(
            context_type=ContextType.EMOTIONAL,
            timestamp=datetime.now(),
            data=data,
            confidence=0.9,
            sources=["mock"],
            relevance_score=1.0
        )

class CognitiveContextProcessor(BaseContextProcessor):
    def process(self, data) -> ContextData:
        return ContextData(
            context_type=ContextType.COGNITIVE,
            timestamp=datetime.now(),
            data=data,
            confidence=0.89,
            sources=["mock"],
            relevance_score=1.0
        )

# Add the new processors to the manager
    def __init__(self):
        self.context_store: Dict[ContextType, List[ContextData]] = {
            ctype: [] for ctype in ContextType
        }
        self.environmental_processor = EnvironmentalContextProcessor()
        self.situational_processor = SituationalContextProcessor()
        self.temporal_processor = TemporalContextProcessor()
        self.social_processor = SocialContextProcessor()
        self.task_processor = TaskContextProcessor()
        self.historical_processor = HistoricalContextProcessor()
        self.emotional_processor = EmotionalContextProcessor()
        self.cognitive_processor = CognitiveContextProcessor()