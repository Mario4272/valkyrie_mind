from datetime import datetime
from context_types import ContextData, ContextType
from base_context_processor import BaseContextProcessor

class SituationalContextProcessor(BaseContextProcessor):
    def process(self, data) -> ContextData:
        return ContextData(
            context_type=ContextType.SITUATIONAL,
            timestamp=datetime.now(),
            data=data,
            confidence=0.85,
            sources=["mock"],
            relevance_score=1.0
        )