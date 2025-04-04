from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import numpy as np
from queue import PriorityQueue
from threading import Lock
import asyncio

from valkyrie_mind.cognitive_subsystems.personality_subsystem import PersonalityCore
from valkyrie_mind.cognitive_subsystems.emotion_manager import EmotionManager
from valkyrie_mind.cognitive_subsystems.relationship_memory import RelationshipMemory
from valkyrie_mind.cognitive_subsystems.value_system import ValueSystem

class CognitiveState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    REFLECTING = "reflecting"
    LEARNING = "learning"
    RESPONDING = "responding"

class ThoughtType(Enum):
    REACTIVE = "reactive"      # Immediate responses
    ANALYTICAL = "analytical"  # Deep thinking
    CREATIVE = "creative"      # Novel solutions
    EMOTIONAL = "emotional"    # Feeling-based
    ETHICAL = "ethical"       # Value-based
    SOCIAL = "social"         # Relationship-based

@dataclass
class Thought:
    type: ThoughtType
    content: Any
    priority: float
    timestamp: datetime
    context: dict
    systems_involved: List[str]

class MindSystem:
    def __init__(self,
                 personality_core: 'PersonalityCore',
                 emotion_manager: 'EmotionManager',
                 relationship_memory: 'RelationshipMemory',
                 value_system: 'ValueSystem'):
        # Core systems
        self.personality = personality_core
        self.emotions = emotion_manager
        self.relationships = relationship_memory
        self.values = value_system
        
        # State management
        self.current_state = CognitiveState.IDLE
        self.state_lock = Lock()
        
        # Thought processing
        self.thought_queue = PriorityQueue()
        self.active_thoughts: List[Thought] = []
        self.thought_history: List[Thought] = []
        
        # Integration parameters
        self.system_weights = self._initialize_system_weights()
        self.integration_thresholds = {
            "minimum_confidence": 0.7,
            "ethical_threshold": 0.9,
            "emotional_sensitivity": 0.6,
            "relationship_influence": 0.5
        }
        
        # Learning and adaptation
        self.learning_rate = 0.1
        self.adaptation_history: List[dict] = []
        
        # Initialize system bridges
        self.system_bridges = self._initialize_system_bridges()
        
    def _initialize_system_weights(self) -> Dict[str, float]:
        """Initialize the relative influence of each system"""
        return {
            "personality": 0.25,
            "emotional": 0.25,
            "relationship": 0.25,
            "values": 0.25
        }

    async def process_input(self, input_data: dict) -> dict:
        """Main entry point for processing any input"""
        try:
            # Generate initial thoughts
            thoughts = self._generate_initial_thoughts(input_data)
            
            # Queue thoughts for processing
            for thought in thoughts:
                await self._queue_thought(thought)
            
            # Process thoughts and generate response
            response = await self._process_thought_queue(input_data)
            
            # Learn from interaction
            await self._learn_from_interaction(input_data, response)
            
            return response
            
        except Exception as e:
            await self._handle_error(e, input_data)
            raise

    async def _process_thought_queue(self, context: dict) -> dict:
        """Process queued thoughts and generate integrated response"""
        responses = []
        
        while not self.thought_queue.empty():
            priority, thought = await self._get_next_thought()
            
            # Process thought through relevant systems
            processed_result = await self._process_thought(thought, context)
            
            if processed_result:
                responses.append(processed_result)
            
            # Update active thoughts
            self.active_thoughts.append(thought)
            
        # Integrate responses
        final_response = await self._integrate_responses(responses, context)
        
        # Archive thoughts
        self.thought_history.extend(self.active_thoughts)
        self.active_thoughts.clear()
        
        return final_response

    async def _process_thought(self, thought: Thought, context: dict) -> Optional[dict]:
        """Process a single thought through relevant systems"""
        processors = {
            ThoughtType.REACTIVE: self._process_reactive_thought,
            ThoughtType.ANALYTICAL: self._process_analytical_thought,
            ThoughtType.CREATIVE: self._process_creative_thought,
            ThoughtType.EMOTIONAL: self._process_emotional_thought,
            ThoughtType.ETHICAL: self._process_ethical_thought,
            ThoughtType.SOCIAL: self._process_social_thought
        }
        
        processor = processors.get(thought.type)
        if processor:
            return await processor(thought, context)
        return None

    async def _integrate_responses(self, 
                                 responses: List[dict], 
                                 context: dict) -> dict:
        """Integrate multiple system responses into coherent output"""
        if not responses:
            return self._generate_default_response(context)
            
        # Gather system states
        personality_state = self.personality.traits
        emotional_state = self.emotions.current_state
        relationship_context = self.relationships.get_relationship_context(context)
        value_guidance = self.values.get_value_guidance(context)
        
        # Weight responses based on system states
        weighted_responses = []
        for response in responses:
            weight = self._calculate_response_weight(
                response,
                personality_state,
                emotional_state,
                relationship_context,
                value_guidance
            )
            weighted_responses.append((weight, response))
            
        # Merge responses
        final_response = await self._merge_weighted_responses(weighted_responses)
        
        # Validate against value system
        if not self._validate_response(final_response):
            return self._generate_safe_response(context)
            
        return final_response

    class SystemBridge:
        """Manages communication between systems"""
        def __init__(self, source_system: str, target_system: str):
            self.source = source_system
            self.target = target_system
            self.transformations: Dict[str, Callable] = {}
            self.cached_data: Dict[str, Any] = {}
            self.last_sync: datetime = datetime.now()

        async def transform_data(self, data: Any, context: dict) -> Any:
            """Transform data from source format to target format"""
            transformation = self.transformations.get(f"{self.source}_to_{self.target}")
            if transformation:
                return await transformation(data, context)
            return data

        def register_transformation(self, name: str, func: Callable) -> None:
            """Register a new data transformation"""
            self.transformations[name] = func

        async def sync_systems(self) -> None:
            """Ensure systems are in sync"""
            self.last_sync = datetime.now()
            # Implement system-specific sync logic

    def _initialize_system_bridges(self) -> Dict[str, 'SystemBridge']:
        """Initialize bridges between systems"""
        bridges = {}
        systems = ['personality', 'emotional', 'relationship', 'values']
        
        for source in systems:
            for target in systems:
                if source != target:
                    bridge_key = f"{source}_to_{target}"
                    bridges[bridge_key] = self.SystemBridge(source, target)
                    
        return bridges

    async def _synchronize_systems(self) -> None:
        """Ensure all systems are in sync"""
        tasks = []
        for bridge in self.system_bridges.values():
            tasks.append(bridge.sync_systems())
        
        await asyncio.gather(*tasks)

    class IntegrationMonitor:
        """Monitors and maintains system integration"""
        def __init__(self):
            self.integration_metrics: Dict[str, float] = {}
            self.system_health: Dict[str, float] = {}
            self.alert_thresholds: Dict[str, float] = {}

        async def monitor_integration(self, 
                                    mind_system: 'MindSystem') -> None:
            """Monitor integration health and performance"""
            while True:
                # Check system health
                await self._check_system_health(mind_system)
                
                # Monitor integration metrics
                await self._update_integration_metrics(mind_system)
                
                # Handle any alerts
                await self._handle_alerts(mind_system)
                
                # Short sleep to prevent excessive CPU usage
                await asyncio.sleep(0.1)

        async def _check_system_health(self, 
                                     mind_system: 'MindSystem') -> None:
            """Check health of individual systems"""
            systems = {
                "personality": mind_system.personality,
                "emotional": mind_system.emotions,
                "relationship": mind_system.relationships,
                "values": mind_system.values
            }
            
            for name, system in systems.items():
                health_score = await self._calculate_system_health(system)
                self.system_health[name] = health_score

        async def _update_integration_metrics(self, 
                                            mind_system: 'MindSystem') -> None:
            """Update metrics on system integration"""
            # Calculate various integration metrics
            response_coherence = await self._calculate_response_coherence(mind_system)
            system_alignment = await self._calculate_system_alignment(mind_system)
            processing_efficiency = await self._calculate_processing_efficiency(mind_system)
            
            self.integration_metrics.update({
                "response_coherence": response_coherence,
                "system_alignment": system_alignment,
                "processing_efficiency": processing_efficiency
            })

    async def run_cognitive_cycle(self) -> None:
        """Main cognitive cycle"""
        monitor = self.IntegrationMonitor()
        
        while True:
            try:
                # Update system state
                with self.state_lock:
                    self._update_cognitive_state()
                
                # Process pending thoughts
                if not self.thought_queue.empty():
                    await self._process_thought_queue({})
                
                # Run system maintenance
                await self._maintain_systems()
                
                # Monitor integration
                await monitor.monitor_integration(self)
                
                # Short sleep to prevent excessive CPU usage
                await asyncio.sleep(0.01)
                
            except Exception as e:
                await self._handle_error(e, {"context": "cognitive_cycle"})

    async def _maintain_systems(self) -> None:
        """Maintain system health and integration"""
        # Sync systems
        await self._synchronize_systems()
        
        # Cleanup old data
        await self._cleanup_old_data()
        
        # Update learning parameters
        await self._update_learning_parameters()
        
        # Optimize system weights
        await self._optimize_system_weights()

    async def _cleanup_old_data(self) -> None:
        """Clean up old thought history and adaptation data"""
        current_time = datetime.now()
        
        # Clean up old thoughts
        self.thought_history = [
            thought for thought in self.thought_history
            if (current_time - thought.timestamp).days < 7
        ]
        
        # Clean up adaptation history
        self.adaptation_history = [
            adaptation for adaptation in self.adaptation_history
            if (current_time - adaptation["timestamp"]).days < 30
        ]

    async def _optimize_system_weights(self) -> None:
        """Optimize the weights of different systems based on performance"""
        if len(self.adaptation_history) < 100:
            return
            
        recent_adaptations = self.adaptation_history[-100:]
        
        # Calculate success rates for each system
        success_rates = {}
        for system in self.system_weights.keys():
            relevant_adaptations = [
                a for a in recent_adaptations
                if a["primary_system"] == system
            ]
            if relevant_adaptations:
                success_rate = sum(a["success_score"] for a in relevant_adaptations) / len(relevant_adaptations)
                success_rates[system] = success_rate
            
        # Adjust weights based on success rates
        total_success = sum(success_rates.values())
        if total_success > 0:
            new_weights = {
                system: rate / total_success
                for system, rate in success_rates.items()
            }
            
            # Smooth the transition to new weights
            for system in self.system_weights:
                current_weight = self.system_weights[system]
                new_weight = new_weights.get(system, current_weight)
                self.system_weights[system] = current_weight * 0.8 + new_weight * 0.2



    def perceive(self, input_data: dict) -> Dict[str, float]:
        """Use personality to modulate perception based on input context."""
        modulation = self.personality.get_response_modulation(input_data)
        return modulation

    def reflect(self) -> None:
        """Allow the personality core to reflect and adjust trait preferences."""
        self.personality.reflect_and_adjust()

    def save_personality_state(self, path: str) -> None:
        self.personality.save_state(path)

    def load_personality_state(self, path: str) -> None:
        self.personality.load_state(path)