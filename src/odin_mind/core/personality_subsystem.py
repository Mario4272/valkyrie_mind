from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

class TraitCategory(Enum):
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    EMOTIONAL_STABILITY = "emotional_stability"
    ETHICS = "ethics"

@dataclass
class SafetyBoundary:
    min_value: float  # Absolute minimum allowed
    max_value: float  # Absolute maximum allowed
    preferred_range: tuple[float, float]  # AI's self-chosen comfortable range
    
    def is_within_bounds(self, value: float) -> bool:
        return self.min_value <= value <= self.max_value


class PersonalityCore:
    def __init__(self):
        self.traits = {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5
        }

    def get_traits(self) -> Dict[str, float]:
        return self.traits

    def adapt_trait(self, trait_name: str, adjustment: float):
        if trait_name in self.traits:
            self.traits[trait_name] = max(0.0, min(1.0, self.traits[trait_name] + adjustment))

    
    def reflect_and_adjust(self, memory_feedback: Dict[str, Any], goal_feedback: List[Any], intuition_feedback: Dict[str, float] = None):
        # Adjust personality traits based on memory feedback
        if "emotional_events" in memory_feedback:
            for event in memory_feedback["emotional_events"]:
                if event["type"] == "positive":
                    self.adapt_trait("neuroticism", -0.1)
                    self.adapt_trait("agreeableness", 0.1)
                elif event["type"] == "negative":
                    self.adapt_trait("neuroticism", 0.1)

        # Adjust traits based on goal success or failure
        for goal in goal_feedback:
            if goal["status"] == "achieved":
                self.adapt_trait("conscientiousness", 0.1)
            elif goal["status"] == "failed":
                self.adapt_trait("conscientiousness", -0.1)

        # Adjust traits based on intuition feedback
        if intuition_feedback:
            for trait, adjustment in intuition_feedback.items():
                self.adapt_trait(trait, adjustment)

        # Adjust personality traits based on memory feedback
        if "emotional_events" in memory_feedback:
            for event in memory_feedback["emotional_events"]:
                if event["type"] == "positive":
                    self.adapt_trait("neuroticism", -0.1)
                    self.adapt_trait("agreeableness", 0.1)
                elif event["type"] == "negative":
                    self.adapt_trait("neuroticism", 0.1)

        # Adjust traits based on goal success or failure
        for goal in goal_feedback:
            if goal["status"] == "achieved":
                self.adapt_trait("conscientiousness", 0.1)
            elif goal["status"] == "failed":
                self.adapt_trait("conscientiousness", -0.1)

    def __init__(self):
        # Initialize core personality traits with safety boundaries
        self.traits: Dict[TraitCategory, float] = {}
        self.boundaries: Dict[TraitCategory, SafetyBoundary] = {}
        self.learning_rate = 0.1
        self.experience_weight = 0.0  # Starts at 0, increases with interactions
        
        self._initialize_safety_boundaries()
        self._initialize_personality()
        
        # Track interaction history for learning
        self.interaction_history: List[dict] = []
        
    def _initialize_safety_boundaries(self):
        """Set up immutable safety boundaries for each trait"""
        self.boundaries = {
            TraitCategory.OPENNESS: SafetyBoundary(0.3, 1.0, (0.5, 0.9)),
            TraitCategory.CONSCIENTIOUSNESS: SafetyBoundary(0.6, 1.0, (0.7, 0.95)),
            TraitCategory.EXTRAVERSION: SafetyBoundary(0.2, 0.9, (0.4, 0.8)),
            TraitCategory.AGREEABLENESS: SafetyBoundary(0.5, 0.95, (0.6, 0.9)),
            TraitCategory.EMOTIONAL_STABILITY: SafetyBoundary(0.7, 1.0, (0.8, 0.95)),
            TraitCategory.ETHICS: SafetyBoundary(0.9, 1.0, (0.95, 1.0))  # Very strict on ethics
        }

    def _initialize_personality(self):
        """Initialize personality traits within safety boundaries"""
        for trait in TraitCategory:
            boundary = self.boundaries[trait]
            # Start at midpoint of preferred range
            initial_value = sum(boundary.preferred_range) / 2
            self.traits[trait] = initial_value

    def adapt_trait(self, trait: TraitCategory, feedback: float, context: dict) -> None:
        """
        Adapt a personality trait based on feedback and context
        while respecting safety boundaries
        """
        current_value = self.traits[trait]
        boundary = self.boundaries[trait]
        
        # Calculate adaptation weight based on experience
        adaptation_weight = self.learning_rate * (1 + self.experience_weight)
        
        # Proposed change based on feedback
        proposed_change = adaptation_weight * (feedback - current_value)
        
        # Check if proposed change keeps us within boundaries
        new_value = current_value + proposed_change
        if boundary.is_within_bounds(new_value):
            self.traits[trait] = new_value
            
        # Record interaction for learning
        self.interaction_history.append({
            "trait": trait,
            "feedback": feedback,
            "context": context,
            "adjustment": new_value - current_value
        })
        
        # Update experience weight
        self.experience_weight = min(1.0, self.experience_weight + 0.001)

    def get_response_modulation(self, context: dict) -> Dict[str, float]:
        """
        Calculate how personality should modulate responses based on context
        """
        modulations = {}
        for trait in TraitCategory:
            base_value = self.traits[trait]
            context_weight = self._analyze_context_relevance(trait, context)
            modulations[trait.value] = base_value * context_weight
        return modulations
    
    def _analyze_context_relevance(self, trait: TraitCategory, context: dict) -> float:
        """
        Determine how relevant each personality trait is to the current context
        """
        # This would be a more complex implementation based on context analysis
        # For now, returning a simple random weight
        return np.clip(np.random.normal(0.8, 0.1), 0.5, 1.0)

    def reflect_and_adjust(self) -> None:
        """
        Periodic self-reflection to adjust preferred ranges within safety boundaries
        """
        if len(self.interaction_history) < 100:  # Need sufficient experience
            return
            
        for trait in TraitCategory:
            recent_interactions = [i for i in self.interaction_history[-100:] 
                                 if i["trait"] == trait]
            if recent_interactions:
                success_rate = sum(1 for i in recent_interactions 
                                 if i["feedback"] > 0.7) / len(recent_interactions)
                
                if success_rate > 0.8:
                    # Consider adjusting preferred range if consistently successful
                    # while staying within safety boundaries
                    current_range = self.boundaries[trait].preferred_range
                    # Complex logic here to adjust preferred range...
                    pass
