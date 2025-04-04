from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import numpy as np

class EmotionalDimension(Enum):
    VALENCE = "valence"  # positive/negative
    AROUSAL = "arousal"  # high/low energy
    DOMINANCE = "dominance"  # in control/controlled
    STABILITY = "stability"  # stable/volatile

@dataclass
class EmotionalState:
    dimensions: Dict[EmotionalDimension, float]
    timestamp: float
    context: dict
    duration: float  # How long this state typically lasts
    intensity: float  # Overall intensity of the emotional state

class EmotionManager:
    def __init__(self, personality_core: 'PersonalityCore'):
        self.personality_core = personality_core
        self.current_state = self._create_baseline_state()
        self.emotional_history: List[EmotionalState] = []
        self.state_duration = 300  # Default state duration in seconds
        
        # Emotion regulation parameters
        self.recovery_rate = 0.1  # Rate of return to baseline
        self.sensitivity = 0.7    # How easily emotions are triggered
        
        # Connect to personality traits
        self.trait_emotion_influences = {
            TraitCategory.EMOTIONAL_STABILITY: 1.5,  # Higher stability = better regulation
            TraitCategory.EXTRAVERSION: 0.8,        # Affects emotional expression
            TraitCategory.AGREEABLENESS: 0.6        # Influences emotional responses
        }

    def _create_baseline_state(self) -> EmotionalState:
        """Create neutral baseline emotional state"""
        return EmotionalState(
            dimensions={dim: 0.5 for dim in EmotionalDimension},
            timestamp=time.time(),
            context={"type": "baseline"},
            duration=self.state_duration,
            intensity=0.5
        )

    def update_emotional_state(self, trigger: dict) -> EmotionalState:
        """
        Update emotional state based on trigger event and personality traits
        """
        # Get personality influence
        personality_modifier = self._calculate_personality_influence()
        
        # Calculate new emotional values
        new_dimensions = {}
        for dimension in EmotionalDimension:
            current_value = self.current_state.dimensions[dimension]
            trigger_value = self._evaluate_trigger_impact(trigger, dimension)
            
            # Apply personality-modified change
            change = (trigger_value - current_value) * self.sensitivity * personality_modifier
            new_value = np.clip(current_value + change, 0.0, 1.0)
            new_dimensions[dimension] = new_value

        # Create new emotional state
        new_state = EmotionalState(
            dimensions=new_dimensions,
            timestamp=time.time(),
            context=trigger,
            duration=self._calculate_duration(trigger),
            intensity=self._calculate_intensity(new_dimensions)
        )

        # Record state change
        self.emotional_history.append(new_state)
        self.current_state = new_state
        
        return new_state

    def _calculate_personality_influence(self) -> float:
        """Calculate how personality traits influence emotional responses"""
        influence = 1.0
        for trait, weight in self.trait_emotion_influences.items():
            trait_value = self.personality_core.traits[trait]
            influence *= (trait_value * weight)
        return np.clip(influence, 0.5, 1.5)

    def _evaluate_trigger_impact(self, trigger: dict, dimension: EmotionalDimension) -> float:
        """Evaluate how a trigger event impacts each emotional dimension"""
        # This would be a more complex implementation based on trigger analysis
        # For now, using a simplified model
        impact_mapping = {
            "positive_interaction": {
                EmotionalDimension.VALENCE: 0.8,
                EmotionalDimension.AROUSAL: 0.6,
                EmotionalDimension.DOMINANCE: 0.7,
                EmotionalDimension.STABILITY: 0.8
            },
            "negative_interaction": {
                EmotionalDimension.VALENCE: 0.2,
                EmotionalDimension.AROUSAL: 0.8,
                EmotionalDimension.DOMINANCE: 0.3,
                EmotionalDimension.STABILITY: 0.4
            },
            # Add more trigger types...
        }
        
        trigger_type = trigger.get("type", "neutral")
        return impact_mapping.get(trigger_type, {}).get(dimension, 0.5)

    def regulate_emotion(self) -> None:
        """
        Gradually return emotional state to baseline based on personality traits
        """
        time_passed = time.time() - self.current_state.timestamp
        if time_passed > self.current_state.duration:
            recovery_strength = self.personality_core.traits[TraitCategory.EMOTIONAL_STABILITY]
            
            new_dimensions = {}
            baseline = self._create_baseline_state().dimensions
            
            for dimension, current_value in self.current_state.dimensions.items():
                baseline_value = baseline[dimension]
                change = (baseline_value - current_value) * self.recovery_rate * recovery_strength
                new_dimensions[dimension] = current_value + change

            self.current_state = EmotionalState(
                dimensions=new_dimensions,
                timestamp=time.time(),
                context={"type": "recovery"},
                duration=self.state_duration,
                intensity=self._calculate_intensity(new_dimensions)
            )

    def _calculate_duration(self, trigger: dict) -> float:
        """Calculate how long an emotional state should last"""
        base_duration = self.state_duration
        intensity = trigger.get("intensity", 0.5)
        importance = trigger.get("importance", 0.5)
        
        # Modify duration based on personality traits
        stability = self.personality_core.traits[TraitCategory.EMOTIONAL_STABILITY]
        
        return base_duration * intensity * importance * (2 - stability)

    def _calculate_intensity(self, dimensions: Dict[EmotionalDimension, float]) -> float:
        """Calculate overall emotional intensity"""
        # Average deviation from neutral (0.5)
        return sum(abs(value - 0.5) for value in dimensions.values()) / len(dimensions)

    def get_response_modulation(self, context: dict) -> Dict[str, float]:
        """
        Calculate how current emotional state should modulate responses
        """
        emotional_influence = {}
        for dimension, value in self.current_state.dimensions.items():
            # Calculate influence based on dimension value and current intensity
            influence = value * self.current_state.intensity
            emotional_influence[dimension.value] = influence
            
        # Combine with personality modulation
        personality_influence = self.personality_core.get_response_modulation(context)
        
        # Blend both influences
        return {
            **emotional_influence,
            **personality_influence
        }
