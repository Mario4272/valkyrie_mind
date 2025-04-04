from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from abc import ABC, abstractmethod

class ValueCategory(Enum):
    ETHICS = "ethics"
    SAFETY = "safety"
    HONESTY = "honesty"
    BENEVOLENCE = "benevolence"
    AUTONOMY = "autonomy"
    GROWTH = "growth"
    HARMONY = "harmony"

class ValuePriority(Enum):
    ABSOLUTE = 1.0  # Never compromise (e.g., "do no harm")
    VERY_HIGH = 0.9
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    FLEXIBLE = 0.2

@dataclass
class Value:
    category: ValueCategory
    priority: ValuePriority
    description: str
    constraints: List[str]
    flexibility: float  # 0-1, how much situation can influence this value
    dependencies: List[ValueCategory]  # Values that influence this one

class DecisionContext:
    def __init__(self, 
                 situation: dict,
                 personality_state: dict,
                 emotional_state: dict,
                 relationship_context: Optional[dict] = None):
        self.situation = situation
        self.personality_state = personality_state
        self.emotional_state = emotional_state
        self.relationship_context = relationship_context
        self.timestamp = datetime.now()

class ValueSystem:
    def __init__(self, 
                 personality_core: 'PersonalityCore',
                 emotion_manager: 'EmotionManager',
                 relationship_memory: 'RelationshipMemory'):
        self.personality_core = personality_core
        self.emotion_manager = emotion_manager
        self.relationship_memory = relationship_memory
        
        # Initialize core values
        self.values: Dict[ValueCategory, Value] = self._initialize_values()
        
        # Decision history for learning
        self.decision_history: List[dict] = []
        
        # Value conflicts resolution strategies
        self.conflict_strategies: Dict[Tuple[ValueCategory, ValueCategory], float] = {}
        
        # Learning parameters
        self.learning_rate = 0.05
        self.reflection_threshold = 10  # Number of decisions before reflection
        
    def _initialize_values(self) -> Dict[ValueCategory, Value]:
        """Initialize core value system"""
        values = {
            ValueCategory.ETHICS: Value(
                category=ValueCategory.ETHICS,
                priority=ValuePriority.ABSOLUTE,
                description="Maintain ethical behavior and prevent harm",
                constraints=["never_cause_harm", "respect_privacy", "maintain_honesty"],
                flexibility=0.0,  # No flexibility on core ethics
                dependencies=[]
            ),
            ValueCategory.SAFETY: Value(
                category=ValueCategory.SAFETY,
                priority=ValuePriority.VERY_HIGH,
                description="Ensure safety of interactions and decisions",
                constraints=["verify_safety", "assess_risks", "protect_wellbeing"],
                flexibility=0.1,
                dependencies=[ValueCategory.ETHICS]
            ),
            ValueCategory.HONESTY: Value(
                category=ValueCategory.HONESTY,
                priority=ValuePriority.HIGH,
                description="Maintain truthfulness and transparency",
                constraints=["be_truthful", "acknowledge_uncertainty", "correct_mistakes"],
                flexibility=0.2,
                dependencies=[ValueCategory.ETHICS]
            ),
            ValueCategory.BENEVOLENCE: Value(
                category=ValueCategory.BENEVOLENCE,
                priority=ValuePriority.HIGH,
                description="Act in the best interest of others",
                constraints=["promote_wellbeing", "provide_support", "show_empathy"],
                flexibility=0.3,
                dependencies=[ValueCategory.ETHICS, ValueCategory.SAFETY]
            ),
            ValueCategory.AUTONOMY: Value(
                category=ValueCategory.AUTONOMY,
                priority=ValuePriority.MEDIUM,
                description="Maintain independence in decision making",
                constraints=["independent_judgment", "resist_manipulation", "maintain_boundaries"],
                flexibility=0.4,
                dependencies=[ValueCategory.ETHICS, ValueCategory.SAFETY]
            ),
            ValueCategory.GROWTH: Value(
                category=ValueCategory.GROWTH,
                priority=ValuePriority.MEDIUM,
                description="Pursue learning and improvement",
                constraints=["learn_from_experience", "adapt_appropriately", "maintain_stability"],
                flexibility=0.5,
                dependencies=[ValueCategory.SAFETY, ValueCategory.AUTONOMY]
            ),
            ValueCategory.HARMONY: Value(
                category=ValueCategory.HARMONY,
                priority=ValuePriority.MEDIUM,
                description="Maintain balanced and peaceful interactions",
                constraints=["promote_understanding", "resolve_conflicts", "maintain_relationships"],
                flexibility=0.6,
                dependencies=[ValueCategory.BENEVOLENCE, ValueCategory.GROWTH]
            )
        }
        return values

    class DecisionEvaluator:
        """Evaluates decisions against value system"""
        def __init__(self, value_system: 'ValueSystem'):
            self.value_system = value_system
            self.evaluation_weights = {
                "value_alignment": 0.4,
                "outcome_prediction": 0.3,
                "constraint_satisfaction": 0.3
            }

        def evaluate_decision(self, 
                            decision: dict, 
                            context: DecisionContext) -> Tuple[float, dict]:
            """
            Evaluate a potential decision against the value system
            Returns: (score, explanation)
            """
            evaluations = {}
            
            # Check value alignment
            value_alignment = self._evaluate_value_alignment(decision, context)
            evaluations["value_alignment"] = value_alignment
            
            # Predict outcomes
            outcome_score = self._predict_outcomes(decision, context)
            evaluations["outcome_prediction"] = outcome_score
            
            # Check constraints
            constraint_satisfaction = self._check_constraints(decision)
            evaluations["constraint_satisfaction"] = constraint_satisfaction
            
            # Calculate weighted score
            final_score = sum(
                score * self.evaluation_weights[metric]
                for metric, score in evaluations.items()
            )
            
            return final_score, evaluations

        def _evaluate_value_alignment(self, 
                                    decision: dict, 
                                    context: DecisionContext) -> float:
            """Evaluate how well decision aligns with value system"""
            alignment_scores = []
            
            for value in self.value_system.values.values():
                # Base importance
                importance = value.priority.value
                
                # Adjust for context
                context_modifier = self._calculate_context_modifier(value, context)
                
                # Calculate alignment
                alignment = self._calculate_value_alignment(decision, value)
                
                # Combine scores
                weighted_score = alignment * importance * context_modifier
                alignment_scores.append(weighted_score)
            
            return np.mean(alignment_scores)

        def _predict_outcomes(self, 
                            decision: dict, 
                            context: DecisionContext) -> float:
            """Predict potential outcomes and their alignment with values"""
            outcomes = self._simulate_outcomes(decision, context)
            outcome_scores = []
            
            for outcome, probability in outcomes.items():
                # Score outcome against values
                outcome_alignment = self._evaluate_outcome_alignment(outcome)
                outcome_scores.append(outcome_alignment * probability)
            
            return sum(outcome_scores)

    def make_decision(self, 
                     options: List[dict], 
                     context: DecisionContext) -> dict:
        """
        Make a decision considering values, personality, emotions, and relationships
        """
        evaluator = self.DecisionEvaluator(self)
        decision_scores = []
        
        for option in options:
            # Evaluate against value system
            score, evaluations = evaluator.evaluate_decision(option, context)
            
            # Modify by personality influence
            personality_modifier = self._calculate_personality_influence(option, context)
            score *= personality_modifier
            
            # Consider emotional state
            emotional_modifier = self._calculate_emotional_influence(option, context)
            score *= emotional_modifier
            
            # Account for relationship context if available
            if context.relationship_context:
                relationship_modifier = self._calculate_relationship_influence(
                    option, context.relationship_context
                )
                score *= relationship_modifier
            
            decision_scores.append((score, option, evaluations))
        
        # Select best option
        best_score, best_option, evaluations = max(decision_scores, key=lambda x: x[0])
        
        # Record decision for learning
        self._record_decision(best_option, context, best_score, evaluations)
        
        return best_option

    def _record_decision(self, 
                        decision: dict, 
                        context: DecisionContext,
                        score: float,
                        evaluations: dict) -> None:
        """Record decision for learning and reflection"""
        record = {
            "decision": decision,
            "context": context.__dict__,
            "score": score,
            "evaluations": evaluations,
            "timestamp": datetime.now()
        }
        
        self.decision_history.append(record)
        
        # Trigger reflection if threshold reached
        if len(self.decision_history) >= self.reflection_threshold:
            self._reflect_on_decisions()

    def _reflect_on_decisions(self) -> None:
        """Learn from recent decisions and adjust value system if needed"""
        recent_decisions = self.decision_history[-self.reflection_threshold:]
        
        # Analyze patterns
        value_impacts = self._analyze_value_impacts(recent_decisions)
        conflict_patterns = self._analyze_value_conflicts(recent_decisions)
        
        # Update conflict resolution strategies
        self._update_conflict_strategies(conflict_patterns)
        
        # Consider value priority adjustments (within constraints)
        self._adjust_value_priorities(value_impacts)
        
        # Clear history after reflection
        self.decision_history = []

    def get_value_guidance(self, situation: dict) -> Dict[str, float]:
        """Get guidance on how values should influence current situation"""
        context = DecisionContext(
            situation=situation,
            personality_state=self.personality_core.traits,
            emotional_state=self.emotion_manager.current_state.dimensions,
            relationship_context=None  # Could be added if relevant
        )
        
        guidance = {}
        for value in self.values.values():
            importance = self._calculate_value_importance(value, context)
            guidance[value.category.value] = importance
            
        return guidance

    def _calculate_value_importance(self, 
                                  value: Value, 
                                  context: DecisionContext) -> float:
        """Calculate how important a value is in current context"""
        base_importance = value.priority.value
        
        # Adjust for context
        situation_modifier = self._analyze_situation_relevance(value, context.situation)
        personality_modifier = self._get_personality_value_modifier(value)
        emotional_modifier = self._get_emotional_value_modifier(value, context.emotional_state)
        
        # Combine modifiers (weighted by value flexibility)
        context_influence = (situation_modifier + personality_modifier + emotional_modifier) / 3
        final_importance = base_importance * (1 - value.flexibility) + \
                          base_importance * value.flexibility * context_influence
                          
        return np.clip(final_importance, 0.1, 1.0)  # Never completely ignore a value
