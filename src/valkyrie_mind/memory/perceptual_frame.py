# valkyrie_graph/perceptual_frame.py

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import uuid

@dataclass
class PerceptualFrame:
    # Unique identifier for the frame
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Timestamp of when the memory was recorded
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Visual input vector (e.g., from CLIP or vision model)
    visual_embedding: Optional[List[float]] = None

    # Audio input vector (e.g., ambient sounds, speech tone)
    auditory_embedding: Optional[List[float]] = None

    # Tactile metadata (e.g., pressure, texture, temperature)
    tactile_data: Optional[Dict[str, float]] = None

    # Tags for recognized smells (symbolic—not numeric)
    olfactory_tags: Optional[List[str]] = None

    # Emotional breakdown of the moment (e.g., joy: 0.7, sadness: 0.2)
    emotion_vector: Optional[Dict[str, float]] = None

    # Intent or motivation behind the experience (e.g., “pick up wife”)
    intent: Optional[str] = None

    # Tags or concepts that symbolically represent this frame
    semantic_tags: Optional[List[str]] = None

    # Links to other frames (before/after/related)
    related_frames: Dict[str, List[str]] = field(default_factory=lambda: {
        "before": [], "after": [], "related": []
    })

    # Salience is what makes this memory rise to the surface:
    # How important, recent, and often it's recalled
    salience: Dict[str, float] = field(default_factory=lambda: {
        "emotional_weight": 0.0,                        # Raw emotional intensity
        "frequency": 0,                                 # How many times it’s been recalled
        "last_accessed": datetime.utcnow().timestamp(), # Timestamp of last recall
        "is_core": False                                # Flag for identity-shaping memories
    })

    def add_relation(self, relation_type: str, frame_id: str):
        # Add a directional connection to another frame under the specified relation type
        if relation_type in self.related_frames:
            self.related_frames[relation_type].append(frame_id)
        else:
            raise ValueError(f"Unknown relation type: {relation_type}")
