from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor

class SensoryType(Enum):
    VISUAL = "visual"
    AUDITORY = "auditory"
    TACTILE = "tactile"
    OLFACTORY = "olfactory"
    GUSTATORY = "gustatory"

class SensoryPriority(Enum):
    CRITICAL = 1.0    # Immediate danger/important signals
    HIGH = 0.8        # Important but not critical
    MEDIUM = 0.6      # Normal input
    LOW = 0.4         # Background information
    MINIMAL = 0.2     # Barely noticeable input

@dataclass
class SensoryInput:
    type: SensoryType
    raw_data: Any
    priority: SensoryPriority
    timestamp: datetime
    metadata: dict
    source_id: str

@dataclass
class ProcessedInput:
    original: SensoryInput
    processed_data: Any
    features: dict
    confidence: float
    related_memories: List[str]
    emotional_tags: List[str]

class SensorySystem:
    def __init__(self, mind_system: 'MindSystem'):
        self.mind = mind_system
        
        # Initialize sensory processors
        self.processors = {
            SensoryType.VISUAL: VisualProcessor(),
            SensoryType.AUDITORY: AuditoryProcessor(),
            SensoryType.TACTILE: TactileProcessor(),
            SensoryType.OLFACTORY: OlfactoryProcessor(),
            SensoryType.GUSTATORY: GustatoryProcessor()
        }
        
        # Input queues for each sense
        self.input_queues = {
            sense: asyncio.Queue() for sense in SensoryType
        }
        
        # Sensory memory (very short term)
        self.sensory_memory = SensoryMemory()
        
        # Cross-modal integration
        self.integration_engine = CrossModalIntegration()
        
        # Processing threads
        self.executor = ThreadPoolExecutor(max_workers=len(SensoryType))
        
        # Attention system
        self.attention = AttentionSystem()
        
        # Calibration parameters
        self.calibration = {sense: {} for sense in SensoryType}
        
    async def process_input(self, input_data: SensoryInput) -> ProcessedInput:
        """Process incoming sensory data"""
        # Queue input
        await self.input_queues[input_data.type].put(input_data)
        
        # Store in sensory memory
        self.sensory_memory.store(input_data)
        
        # Get processor for this type
        processor = self.processors[input_data.type]
        
        # Process based on priority
        if input_data.priority in [SensoryPriority.CRITICAL, SensoryPriority.HIGH]:
            # Immediate processing
            processed = await processor.process_urgent(input_data)
        else:
            # Normal processing
            processed = await processor.process(input_data)
            
        # Integrate with other senses
        integrated_data = await self.integration_engine.integrate(
            processed,
            self.sensory_memory.get_recent(timespan_ms=100)
        )
        
        # Update attention system
        self.attention.update(integrated_data)
        
        return integrated_data

class SensoryProcessor(ABC):
    """Base class for sensory processors"""
    def __init__(self):
        self.processing_pipeline: List[Callable] = []
        self.feature_extractors: Dict[str, Callable] = {}
        self.calibration_data: dict = {}
        
    @abstractmethod
    async def process(self, input_data: SensoryInput) -> ProcessedInput:
        pass
        
    @abstractmethod
    async def process_urgent(self, input_data: SensoryInput) -> ProcessedInput:
        pass
        
    @abstractmethod
    async def calibrate(self) -> None:
        pass

class VisualProcessor(SensoryProcessor):
    """Processes visual input"""
    def __init__(self):
        super().__init__()
        self.resolution = (1920, 1080)  # Default resolution
        self.color_calibration = {}
        self.motion_detector = MotionDetector()
        self.object_recognizer = ObjectRecognizer()
        self.spatial_analyzer = SpatialAnalyzer()
        
    async def process(self, input_data: SensoryInput) -> ProcessedInput:
        # Basic image processing
        normalized = await self._normalize_image(input_data.raw_data)
        
        # Feature extraction
        features = {
            "objects": await self.object_recognizer.detect(normalized),
            "motion": await self.motion_detector.analyze(normalized),
            "spatial": await self.spatial_analyzer.analyze(normalized)
        }
        
        # Create processed input
        return ProcessedInput(
            original=input_data,
            processed_data=normalized,
            features=features,
            confidence=self._calculate_confidence(features),
            related_memories=[],  # To be filled by memory system
            emotional_tags=[]     # To be filled by emotional system
        )

class AuditoryProcessor(SensoryProcessor):
    """Processes auditory input"""
    def __init__(self):
        super().__init__()
        self.frequency_analyzer = FrequencyAnalyzer()
        self.speech_recognizer = SpeechRecognizer()
        self.sound_classifier = SoundClassifier()
        
    async def process(self, input_data: SensoryInput) -> ProcessedInput:
        # Basic audio processing
        normalized = await self._normalize_audio(input_data.raw_data)
        
        # Feature extraction
        features = {
            "frequency_profile": await self.frequency_analyzer.analyze(normalized),
            "speech_content": await self.speech_recognizer.process(normalized),
            "sound_type": await self.sound_classifier.classify(normalized)
        }
        
        return ProcessedInput(
            original=input_data,
            processed_data=normalized,
            features=features,
            confidence=self._calculate_confidence(features),
            related_memories=[],
            emotional_tags=[]
        )

class TactileProcessor(SensoryProcessor):
    """Processes touch and pressure input"""
    def __init__(self):
        super().__init__()
        self.pressure_analyzer = PressureAnalyzer()
        self.texture_analyzer = TextureAnalyzer()
        self.temperature_analyzer = TemperatureAnalyzer()
        
    async def process(self, input_data: SensoryInput) -> ProcessedInput:
        features = {
            "pressure": await self.pressure_analyzer.analyze(input_data.raw_data),
            "texture": await self.texture_analyzer.analyze(input_data.raw_data),
            "temperature": await self.temperature_analyzer.analyze(input_data.raw_data)
        }
        
        return ProcessedInput(
            original=input_data,
            processed_data=input_data.raw_data,
            features=features,
            confidence=self._calculate_confidence(features),
            related_memories=[],
            emotional_tags=[]
        )

class CrossModalIntegration:
    """Integrates information across different senses"""
    def __init__(self):
        self.integration_weights = {
            SensoryType.VISUAL: 0.3,
            SensoryType.AUDITORY: 0.3,
            SensoryType.TACTILE: 0.2,
            SensoryType.OLFACTORY: 0.1,
            SensoryType.GUSTATORY: 0.1
        }
        
        self.correlation_detector = ModalityCorrelationDetector()
        self.conflict_resolver = ConflictResolver()
        
    async def integrate(self, 
                       new_input: ProcessedInput, 
                       recent_inputs: List[ProcessedInput]) -> ProcessedInput:
        """Integrate new input with recent inputs from other senses"""
        # Find correlations
        correlations = await self.correlation_detector.find_correlations(
            new_input, recent_inputs
        )
        
        # Resolve any conflicts
        resolved_data = await self.conflict_resolver.resolve(
            new_input, correlations
        )
        
        # Enhance features based on cross-modal information
        enhanced_features = await self._enhance_features(
            resolved_data, correlations
        )
        
        # Update processed input with enhanced features
        resolved_data.features.update(enhanced_features)
        
        return resolved_data

class AttentionSystem:
    """Manages attention and priority of sensory processing"""
    def __init__(self):
        self.current_focus: Optional[SensoryType] = None
        self.attention_weights = {sense: 0.2 for sense in SensoryType}
        self.priority_threshold = 0.7
        
    def update(self, new_input: ProcessedInput) -> None:
        """Update attention based on new input"""
        # Calculate attention score
        score = self._calculate_attention_score(new_input)
        
        # Update weights
        self._update_weights(new_input.original.type, score)
        
        # Set focus if high priority
        if score > self.priority_threshold:
            self.current_focus = new_input.original.type

    def _calculate_attention_score(self, input_data: ProcessedInput) -> float:
        """Calculate how much attention this input should receive"""
        base_score = input_data.original.priority.value
        
        # Modify based on features
        novelty_score = self._calculate_novelty(input_data)
        relevance_score = self._calculate_relevance(input_data)
        urgency_score = self._calculate_urgency(input_data)
        
        # Weighted combination
        return (base_score * 0.4 + 
                novelty_score * 0.2 +
                relevance_score * 0.2 +
                urgency_score * 0.2)

class SensoryMemory:
    """Very short-term sensory memory"""
    def __init__(self):
        self.memory_duration_ms = 1000  # 1 second
        self.memories: Dict[SensoryType, List[Tuple[datetime, ProcessedInput]]] = {
            sense: [] for sense in SensoryType
        }
        
    def store(self, input_data: SensoryInput) -> None:
        """Store new sensory input"""
        self.memories[input_data.type].append(
            (datetime.now(), input_data)
        )
        self._cleanup()
        
    def get_recent(self, timespan_ms: int = None) -> List[ProcessedInput]:
        """Get recent memories within timespan"""
        if timespan_ms is None:
            timespan_ms = self.memory_duration_ms
            
        cutoff_time = datetime.now() - timedelta(milliseconds=timespan_ms)
        
        recent_memories = []
        for sense_memories in self.memories.values():
            recent_memories.extend([
                memory for timestamp, memory in sense_memories
                if timestamp >= cutoff_time
            ])
            
        return recent_memories
        
    def _cleanup(self) -> None:
        """Remove old memories"""
        cutoff_time = datetime.now() - timedelta(milliseconds=self.memory_duration_ms)
        
        for sense in self.memories:
            self.memories[sense] = [
                (timestamp, memory) 
                for timestamp, memory in self.memories[sense]
                if timestamp >= cutoff_time
            ]

# Shared Sensory System Functionality

from enum import Enum
from abc import ABC, abstractmethod
from typing import List

# Enums for Sensory Types
class SensoryType(Enum):
    AUDITORY = "auditory"
    VISUAL = "visual"
    TACTILE = "tactile"
    GUSTATORY = "gustatory"
    OLFACTORY = "olfactory"

class SensoryPriority(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

# Base class for Sensory Processors
class SensoryProcessor(ABC):
    @abstractmethod
    def process_input(self, sensory_input: Any) -> Any:
        pass

# Sensory Memory System
class SensoryMemory:
    def __init__(self):
        self.memory = []

    def store(self, input_data: Any):
        self.memory.append(input_data)

    def retrieve(self, filter_func=None) -> List[Any]:
        if filter_func:
            return [data for data in self.memory if filter_func(data)]
        return self.memory

# Attention System
class AttentionSystem:
    def __init__(self):
        self.focus = None

    def update_focus(self, sensory_type: SensoryType):
        self.focus = sensory_type

    def get_focus(self) -> SensoryType:
        return self.focus
