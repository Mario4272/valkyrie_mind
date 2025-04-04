from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import numpy as np

class RelationType(Enum):
    STRANGER = "stranger"
    ACQUAINTANCE = "acquaintance"
    FAMILIAR = "familiar"
    CLOSE = "close"

class InteractionType(Enum):
    CASUAL = "casual"
    PROFESSIONAL = "professional"
    PERSONAL = "personal"
    EMOTIONAL = "emotional"
    CONFLICT = "conflict"
    COLLABORATION = "collaboration"

@dataclass
class Interaction:
    timestamp: datetime
    type: InteractionType
    context: dict
    emotional_state: Dict[str, float]
    outcome_score: float  # -1 to 1
    topics: Set[str]
    duration: timedelta

@dataclass
class RelationshipMetrics:
    trust: float  # 0-1
    rapport: float  # 0-1
    understanding: float  # 0-1
    compatibility: float  # 0-1
    interaction_frequency: float  # interactions per time period
    last_interaction: datetime

class RelationshipMemory:
    def __init__(self, personality_core: 'PersonalityCore', emotion_manager: 'EmotionManager'):
        self.personality_core = personality_core
        self.emotion_manager = emotion_manager
        self.relationships: Dict[str, 'Relationship'] = {}
        self.global_patterns: Dict[str, float] = {}
        
        # Learning parameters
        self.learning_rate = 0.1
        self.memory_decay = 0.99  # How quickly old interactions lose influence
        
    def get_or_create_relationship(self, person_id: str) -> 'Relationship':
        """Get existing relationship or create new one"""
        if person_id not in self.relationships:
            self.relationships[person_id] = Relationship(
                person_id=person_id,
                personality_core=self.personality_core,
                emotion_manager=self.emotion_manager
            )
        return self.relationships[person_id]

class Relationship:
    def __init__(self, person_id: str, personality_core: 'PersonalityCore', 
                 emotion_manager: 'EmotionManager'):
        self.person_id = person_id
        self.personality_core = personality_core
        self.emotion_manager = emotion_manager
        
        # Initialize relationship state
        self.type = RelationType.STRANGER
        self.metrics = RelationshipMetrics(
            trust=0.3,
            rapport=0.2,
            understanding=0.1,
            compatibility=0.5,
            interaction_frequency=0.0,
            last_interaction=datetime.now()
        )
        
        # History tracking
        self.interactions: List[Interaction] = []
        self.topics_discussed: Set[str] = set()
        self.shared_experiences: List[dict] = []
        self.communication_preferences: Dict[str, float] = {}
        
        # Behavioral adaptations
        self.personal_boundaries: Dict[str, float] = {}
        self.interaction_patterns: Dict[str, float] = {}
        
    def record_interaction(self, interaction_type: InteractionType, 
                         context: dict, outcome: float) -> None:
        """Record a new interaction and update relationship metrics"""
        current_emotion = self.emotion_manager.current_state.dimensions
        
        interaction = Interaction(
            timestamp=datetime.now(),
            type=interaction_type,
            context=context,
            emotional_state=current_emotion,
            outcome_score=outcome,
            topics=set(context.get("topics", [])),
            duration=timedelta(seconds=context.get("duration", 0))
        )
        
        self.interactions.append(interaction)
        self.topics_discussed.update(interaction.topics)
        
        # Update metrics based on interaction
        self._update_metrics(interaction)
        self._update_relationship_type()
        self._learn_from_interaction(interaction)

    def _update_metrics(self, interaction: Interaction) -> None:
        """Update relationship metrics based on new interaction"""
        # Calculate time-weighted outcome
        time_since_last = datetime.now() - self.metrics.last_interaction
        time_weight = np.exp(-time_since_last.days / 30)  # Exponential decay
        
        # Update trust
        trust_impact = interaction.outcome_score * self._calculate_trust_modifier(interaction)
        self.metrics.trust = np.clip(
            self.metrics.trust + trust_impact * self.personality_core.learning_rate,
            0, 1
        )
        
        # Update rapport
        rapport_change = self._calculate_rapport_change(interaction)
        self.metrics.rapport = np.clip(
            self.metrics.rapport + rapport_change,
            0, 1
        )
        
        # Update understanding
        understanding_change = len(interaction.topics) * 0.05
        self.metrics.understanding = np.clip(
            self.metrics.understanding + understanding_change,
            0, 1
        )
        
        # Update compatibility
        self.metrics.compatibility = self._calculate_compatibility()
        
        # Update frequency
        self.metrics.interaction_frequency = self._calculate_frequency()
        self.metrics.last_interaction = datetime.now()

    def _calculate_trust_modifier(self, interaction: Interaction) -> float:
        """Calculate how much an interaction should affect trust"""
        base_modifier = {
            InteractionType.CASUAL: 0.1,
            InteractionType.PROFESSIONAL: 0.2,
            InteractionType.PERSONAL: 0.3,
            InteractionType.EMOTIONAL: 0.4,
            InteractionType.CONFLICT: -0.3,
            InteractionType.COLLABORATION: 0.3
        }[interaction.type]
        
        # Modify based on emotional state
        emotional_intensity = sum(interaction.emotional_state.values()) / len(interaction.emotional_state)
        return base_modifier * (1 + emotional_intensity)

    def get_interaction_guidance(self, context: dict) -> Dict[str, float]:
        """Get guidance for how to interact based on relationship history"""
        relationship_influence = {
            "trust_level": self.metrics.trust,
            "rapport_level": self.metrics.rapport,
            "understanding_level": self.metrics.understanding,
            "relationship_type": self.type.value
        }
        
        # Combine with personality and emotional influences
        personality_influence = self.personality_core.get_response_modulation(context)
        emotional_influence = self.emotion_manager.get_response_modulation(context)
        
        # Weighted combination based on relationship type
        weights = self._get_influence_weights()
        
        return {
            "relationship": relationship_influence,
            "personality": personality_influence,
            "emotional": emotional_influence,
            "weights": weights
        }

    def _get_influence_weights(self) -> Dict[str, float]:
        """Calculate how much each system should influence interactions"""
        relationship_weight = min(self.metrics.trust + self.metrics.rapport, 1.0)
        
        return {
            "relationship": relationship_weight,
            "personality": 1.0 - (relationship_weight * 0.5),  # Personality remains somewhat stable
            "emotional": self.emotion_manager.current_state.intensity
        }

    def _learn_from_interaction(self, interaction: Interaction) -> None:
        """Learn and adapt from interaction experience"""
        # Update communication preferences
        for topic in interaction.topics:
            if topic not in self.communication_preferences:
                self.communication_preferences[topic] = 0.5
            
            # Adjust preference based on outcome
            current_pref = self.communication_preferences[topic]
            self.communication_preferences[topic] = np.clip(
                current_pref + (interaction.outcome_score * 0.1),
                0, 1
            )

        # Update boundaries based on emotional response
        for dimension, value in interaction.emotional_state.items():
            if dimension not in self.personal_boundaries:
                self.personal_boundaries[dimension] = 0.5
            
            # Adjust boundaries based on emotional comfort
            comfort_level = 1 - abs(value - 0.5)  # How comfortable the emotion was
            current_boundary = self.personal_boundaries[dimension]
            self.personal_boundaries[dimension] = np.clip(
                current_boundary + (comfort_level - 0.5) * 0.1,
                0, 1
            )

    def _update_relationship_type(self) -> None:
        """Update relationship type based on metrics"""
        if self.metrics.trust > 0.8 and self.metrics.rapport > 0.8:
            self.type = RelationType.CLOSE
        elif self.metrics.trust > 0.6 and self.metrics.rapport > 0.5:
            self.type = RelationType.FAMILIAR
        elif self.metrics.trust > 0.4:
            self.type = RelationType.ACQUAINTANCE
        else:
            self.type = RelationType.STRANGER

    def _calculate_compatibility(self) -> float:
        """Calculate overall compatibility score"""
        if not self.interactions:
            return 0.5
            
        recent_interactions = sorted(self.interactions, key=lambda x: x.timestamp)[-10:]
        return np.mean([i.outcome_score for i in recent_interactions])

    def _calculate_frequency(self) -> float:
        """Calculate interaction frequency (interactions per week)"""
        if len(self.interactions) < 2:
            return 0.0
            
        time_span = self.interactions[-1].timestamp - self.interactions[0].timestamp
        weeks = time_span.days / 7 or 1  # Avoid division by zero
        return len(self.interactions) / weeks
