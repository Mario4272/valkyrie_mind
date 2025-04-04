from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
import asyncio
from datetime import datetime
import uuid
import numpy as np

class GoalStatus(Enum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    BLOCKED = "blocked"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"

class GoalType(Enum):
    ACHIEVEMENT = "achievement"  # One-time accomplishment
    MAINTENANCE = "maintenance"  # Ongoing state maintenance
    APPROACH = "approach"       # Gradual progress toward target
    AVOIDANCE = "avoidance"    # Staying away from state/condition
    LEARNING = "learning"       # Knowledge/skill acquisition
    OPTIMIZATION = "optimization"  # Improving performance metrics

class GoalPriority(Enum):
    CRITICAL = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    BACKGROUND = auto()

class GoalTimeframe(Enum):
    IMMEDIATE = "immediate"     # Now
    SHORT_TERM = "short_term"   # Hours to days
    MEDIUM_TERM = "medium_term" # Days to weeks
    LONG_TERM = "long_term"     # Weeks to months
    INDEFINITE = "indefinite"   # Ongoing

@dataclass
class GoalContext:
    """Context information for a goal"""
    environmental_state: dict
    system_state: dict
    resource_availability: Dict[str, float]
    constraints: List[str]
    dependencies: List[str]
    conflicts: List[str]
    priority_factors: Dict[str, float]

@dataclass
class GoalMetrics:
    """Metrics for measuring goal progress"""
    progress: float  # 0-1
    success_probability: float
    resource_usage: Dict[str, float]
    time_remaining: Optional[float]
    blockers: List[str]
    dependencies_met: bool
    performance_metrics: Dict[str, float]

@dataclass
class Goal:
    """Represents a system goal"""
    goal_id: str
    parent_id: Optional[str]
    type: GoalType
    description: str
    priority: GoalPriority
    timeframe: GoalTimeframe
    creation_time: datetime
    deadline: Optional[datetime]
    status: GoalStatus
    context: GoalContext
    metrics: GoalMetrics
    subgoals: List[str]  # subgoal IDs
    completion_criteria: Dict[str, Any]
    adaptation_rules: Dict[str, Any]
    metadata: Dict[str, Any]

class PlanStep:
    """Represents a step in a plan"""
    def __init__(self, 
                 step_id: str,
                 action: str,
                 preconditions: List[str],
                 postconditions: List[str],
                 estimated_duration: float,
                 resource_requirements: Dict[str, float],
                 success_probability: float):
        self.step_id = step_id
        self.action = action
        self.preconditions = preconditions
        self.postconditions = postconditions
        self.estimated_duration = estimated_duration
        self.resource_requirements = resource_requirements
        self.success_probability = success_probability
        self.status = "pending"
        self.actual_duration: Optional[float] = None
        self.actual_resources: Optional[Dict[str, float]] = None
        self.result: Optional[dict] = None

@dataclass
class Plan:
    """Represents a plan to achieve a goal"""
    plan_id: str
    goal_id: str
    steps: List[PlanStep]
    total_duration: float
    total_resources: Dict[str, float]
    success_probability: float
    alternative_plans: List[str]  # IDs of alternative plans
    adaptation_history: List[dict]
    current_step_index: int = 0

class GoalManager:
    """Manages the goal hierarchy and lifecycle"""
    def __init__(self, mind_system: 'MindSystem'):
        self.mind = mind_system
        self.goals: Dict[str, Goal] = {}
        self.goal_hierarchy: Dict[str, List[str]] = {}  # parent_id -> child_ids
        self.active_goals: Set[str] = set()
        self.goal_history: List[dict] = []
        
        # Planning components
        self.planner = Planner()
        self.scheduler = Scheduler()
        self.resource_manager = ResourceManager()
        
        # Monitoring components
        self.progress_monitor = ProgressMonitor()
        self.conflict_detector = ConflictDetector()
        
        # Adaptation components
        self.goal_adapter = GoalAdapter()
        self.strategy_learner = StrategyLearner()

    async def create_goal(self, 
                         goal_type: GoalType,
                         description: str,
                         priority: GoalPriority,
                         timeframe: GoalTimeframe,
                         completion_criteria: Dict[str, Any],
                         parent_id: Optional[str] = None) -> str:
        """Create a new goal"""
        goal_id = str(uuid.uuid4())
        
        # Create goal context
        context = await self._create_goal_context(parent_id)
        
        # Initialize metrics
        metrics = GoalMetrics(
            progress=0.0,
            success_probability=1.0,
            resource_usage={},
            time_remaining=None,
            blockers=[],
            dependencies_met=True,
            performance_metrics={}
        )
        
        # Create goal
        goal = Goal(
            goal_id=goal_id,
            parent_id=parent_id,
            type=goal_type,
            description=description,
            priority=priority,
            timeframe=timeframe,
            creation_time=datetime.now(),
            deadline=None,  # Set based on timeframe
            status=GoalStatus.PROPOSED,
            context=context,
            metrics=metrics,
            subgoals=[],
            completion_criteria=completion_criteria,
            adaptation_rules={},
            metadata={}
        )
        
        # Store goal
        self.goals[goal_id] = goal
        
        # Update hierarchy
        if parent_id:
            if parent_id not in self.goal_hierarchy:
                self.goal_hierarchy[parent_id] = []
            self.goal_hierarchy[parent_id].append(goal_id)
        
        # Generate initial plan
        await self.planner.create_plan(goal)
        
        return goal_id

    async def activate_goal(self, goal_id: str) -> None:
        """Activate a goal and begin execution"""
        if goal_id not in self.goals:
            raise ValueError(f"Goal {goal_id} not found")
            
        goal = self.goals[goal_id]
        
        # Check for conflicts
        conflicts = await self.conflict_detector.detect_conflicts(goal)
        if conflicts:
            goal.status = GoalStatus.BLOCKED
            goal.metrics.blockers.extend(conflicts)
            return
            
        # Check resource availability
        if not await self.resource_manager.check_resources(goal):
            goal.status = GoalStatus.BLOCKED
            goal.metrics.blockers.append("insufficient_resources")
            return
            
        # Activate goal
        goal.status = GoalStatus.ACTIVE
        self.active_goals.add(goal_id)
        
        # Start monitoring
        await self.progress_monitor.start_monitoring(goal)
        
        # Begin execution
        await self._execute_goal(goal)

    async def _execute_goal(self, goal: Goal) -> None:
        """Execute a goal's plan"""
        plan = await self.planner.get_plan(goal.goal_id)
        if not plan:
            raise RuntimeError(f"No plan found for goal {goal.goal_id}")
            
        while plan.current_step_index < len(plan.steps):
            step = plan.steps[plan.current_step_index]
            
            # Execute step
            try:
                success = await self._execute_step(step)
                if not success:
                    # Adapt plan
                    await self._handle_step_failure(goal, plan, step)
                    continue
                    
                plan.current_step_index += 1
                
            except Exception as e:
                await self._handle_execution_error(goal, plan, step, e)
                return
                
        # Check completion
        if await self._check_completion(goal):
            goal.status = GoalStatus.COMPLETED
            self.active_goals.remove(goal.goal_id)

class Planner:
    """Generates and manages plans for goals"""
    def __init__(self):
        self.plans: Dict[str, Plan] = {}
        self.action_templates: Dict[str, dict] = {}
        self.planning_strategies: Dict[GoalType, Callable] = {}
        
    async def create_plan(self, goal: Goal) -> Plan:
        """Create a plan for a goal"""
        # Select planning strategy
        strategy = self.planning_strategies.get(goal.type, self._default_planning_strategy)
        
        # Generate plan steps
        steps = await strategy(goal)
        
        # Calculate plan metrics
        total_duration = sum(step.estimated_duration for step in steps)
        total_resources = self._aggregate_resources([step.resource_requirements for step in steps])
        success_probability = np.prod([step.success_probability for step in steps])
        
        # Create plan
        plan = Plan(
            plan_id=str(uuid.uuid4()),
            goal_id=goal.goal_id,
            steps=steps,
            total_duration=total_duration,
            total_resources=total_resources,
            success_probability=success_probability,
            alternative_plans=[],
            adaptation_history=[]
        )
        
        self.plans[plan.plan_id] = plan
        return plan
    
    async def adapt_plan(self, plan: Plan, feedback: dict) -> Plan:
        """Adapt a plan based on execution feedback"""
        # Create adaptation record
        adaptation = {
            'timestamp': datetime.now(),
            'trigger': feedback,
            'original_plan': plan
        }
        
        # Generate adapted plan
        adapted_steps = await self._adapt_steps(plan.steps, feedback)
        
        # Create new plan version
        adapted_plan = Plan(
            plan_id=str(uuid.uuid4()),
            goal_id=plan.goal_id,
            steps=adapted_steps,
            total_duration=sum(step.estimated_duration for step in adapted_steps),
            total_resources=self._aggregate_resources([step.resource_requirements for step in adapted_steps]),
            success_probability=np.prod([step.success_probability for step in adapted_steps]),
            alternative_plans=plan.alternative_plans + [plan.plan_id],
            adaptation_history=plan.adaptation_history + [adaptation]
        )
        
        self.plans[adapted_plan.plan_id] = adapted_plan
        return adapted_plan

class ProgressMonitor:
    """Monitors goal progress and triggers adaptations"""
    def __init__(self):
        self.monitored_goals: Dict[str, asyncio.Task] = {}
        self.progress_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.alert_thresholds = {
            'progress_rate': 0.1,
            'resource_usage': 0.8,
            'time_remaining': 0.2
        }
        
    async def start_monitoring(self, goal: Goal) -> None:
        """Start monitoring a goal"""
        if goal.goal_id in self.monitored_goals:
            return
            
        self.progress_history[goal.goal_id] = []
        monitoring_task = asyncio.create_task(self._monitor_goal(goal))
        self.monitored_goals[goal.goal_id] = monitoring_task
        
    async def _monitor_goal(self, goal: Goal) -> None:
        """Monitor a single goal"""
        while goal.status in {GoalStatus.ACTIVE, GoalStatus.IN_PROGRESS}:
            # Update metrics
            await self._update_metrics(goal)
            
            # Check for issues
            issues = await self._check_issues(goal)
            if issues:
                await self._handle_issues(goal, issues)
                
            # Record progress
            self.progress_history[goal.goal_id].append(
                (datetime.now(), goal.metrics.progress)
            )
            
            await asyncio.sleep(1)  # Adjust monitoring frequency as needed

class GoalAdapter:
    """Adapts goals based on execution feedback"""
    def __init__(self):
        self.adaptation_strategies: Dict[str, Callable] = {}
        self.adaptation_history: List[dict] = []
        
    async def adapt_goal(self, goal: Goal, trigger: dict) -> Goal:
        """Adapt a goal based on trigger conditions"""
        # Create adaptation record
        adaptation = {
            'timestamp': datetime.now(),
            'goal_id': goal.goal_id,
            'trigger': trigger,
            'original_state': goal
        }
        
        # Select adaptation strategy
        strategy = self._select_strategy(goal, trigger)
        
        # Apply adaptation
        adapted_goal = await strategy(goal)
        
        # Record adaptation
        adaptation['adapted_state'] = adapted_goal
        self.adaptation_history.append(adaptation)
        
        return adapted_goal

    async def _select_strategy(self, goal: Goal, trigger: dict) -> Callable:
        """Select appropriate adaptation strategy"""
        if trigger.get('type') == 'resource_constraint':
            return self.adaptation_strategies.get('resource_adaptation')
        elif trigger.get('type') == 'time_constraint':
            return self.adaptation_strategies.get('time_adaptation')
        elif trigger.get('type') == 'priority_change':
            return self.adaptation_strategies.get('priority_adaptation')
        else:
            return self.adaptation_strategies.get('default_adaptation')

class StrategyLearner:
    """Learns and improves goal achievement strategies"""
    def __init__(self):
        self.strategy_patterns: Dict[str, List[dict]] = {}
        self.success_metrics: Dict[str, List[float]] = {}
        self.learning_rate = 0.1
        
    async def learn_from_execution(self, goal: Goal, execution_history: List[dict]) -> None:
        """Learn from goal execution history"""
        # Extract patterns
        patterns = await self._extract_patterns(execution_history)
        
        # Update strategy patterns
        await self._update_patterns(goal.type, patterns)
        
        # Update success metrics
        await self._update_metrics(goal, execution_history)
        
        # Generate new strategies
        await self._generate_strategies(goal.type)

class Scheduler:
    """Schedules goal execution and manages dependencies"""
    def __init__(self):
        self.schedule: Dict[str, List[Tuple[datetime, str]]] = {}  # timeframe -> [(time, goal_id)]
        self.dependencies: Dict[str, Set[str]] = {}  # goal_id -> dependent_goal_ids
        
    async def schedule_goal(self, goal: Goal) -> None:
        """Schedule a goal for execution"""
        # Check dependencies
        dependencies = self._get_dependencies(goal)
        if dependencies:
            self.dependencies[goal.goal_id] = dependencies
            if not await self._are_dependencies_met(dependencies):
                goal.status = GoalStatus.BLOCKED
                goal.metrics.dependencies_met = False
                return
                
        # Calculate execution time
        execution_time = await self._calculate_execution_time(goal)
        # Add to schedule
        if goal.timeframe not in self.schedule:
            self.schedule[goal.timeframe] = []
            
        self.schedule[goal.timeframe].append((execution_time, goal.goal_id))
        
        # Sort schedule
        self.schedule[goal.timeframe].sort(key=lambda x: x[0])
        
    async def _calculate_execution_time(self, goal: Goal) -> datetime:
        """Calculate when a goal should be executed"""
        now = datetime.now()
        
        if goal.timeframe == GoalTimeframe.IMMEDIATE:
            return now
        elif goal.timeframe == GoalTimeframe.SHORT_TERM:
            return now + timedelta(days=1)
        elif goal.timeframe == GoalTimeframe.MEDIUM_TERM:
            return now + timedelta(weeks=1)
        elif goal.timeframe == GoalTimeframe.LONG_TERM:
            return now + timedelta(months=1)
        else:  # INDEFINITE
            return now + timedelta(years=1)

    async def _are_dependencies_met(self, dependencies: Set[str]) -> bool:
        """Check if all dependencies are met"""
        for dep_id in dependencies:
            if dep_id not in self.goals:
                return False
            if self.goals[dep_id].status != GoalStatus.COMPLETED:
                return False
        return True

    def _get_dependencies(self, goal: Goal) -> Set[str]:
        """Get all dependencies for a goal"""
        dependencies = set()
        for dep in goal.context.dependencies:
            dependencies.add(dep)
            # Add transitive dependencies
            if dep in self.dependencies:
                dependencies.update(self.dependencies[dep])
        return dependencies

class ResourceManager:
    """Manages resource allocation and tracking for goals"""
    def __init__(self):
        self.available_resources: Dict[str, float] = {}
        self.allocated_resources: Dict[str, Dict[str, float]] = {}  # goal_id -> {resource -> amount}
        self.resource_history: List[dict] = []
        self.reservation_threshold = 0.9
        
    async def check_resources(self, goal: Goal) -> bool:
        """Check if required resources are available"""
        plan = await self.planner.get_plan(goal.goal_id)
        if not plan:
            return False
            
        required_resources = plan.total_resources
        
        # Check each resource
        for resource, amount in required_resources.items():
            if resource not in self.available_resources:
                return False
            
            # Calculate total allocated
            total_allocated = sum(
                alloc.get(resource, 0)
                for alloc in self.allocated_resources.values()
            )
            
            # Check availability
            if (total_allocated + amount) > (self.available_resources[resource] * self.reservation_threshold):
                return False
                
        return True
        
    async def allocate_resources(self, goal: Goal) -> bool:
        """Allocate resources for a goal"""
        plan = await self.planner.get_plan(goal.goal_id)
        if not plan:
            return False
            
        required_resources = plan.total_resources
        
        # Try to allocate
        if await self.check_resources(goal):
            self.allocated_resources[goal.goal_id] = required_resources
            
            # Record allocation
            self.resource_history.append({
                'timestamp': datetime.now(),
                'goal_id': goal.goal_id,
                'allocation': required_resources.copy()
            })
            
            return True
            
        return False
        
    async def release_resources(self, goal_id: str) -> None:
        """Release resources allocated to a goal"""
        if goal_id in self.allocated_resources:
            released = self.allocated_resources.pop(goal_id)
            
            # Record release
            self.resource_history.append({
                'timestamp': datetime.now(),
                'goal_id': goal_id,
                'release': released
            })

class ConflictDetector:
    """Detects and manages conflicts between goals"""
    def __init__(self):
        self.conflict_patterns: Dict[str, dict] = {}
        self.active_conflicts: Dict[str, Set[str]] = {}  # goal_id -> conflicting_goal_ids
        self.resolution_strategies: Dict[str, Callable] = {}
        
    async def detect_conflicts(self, goal: Goal) -> List[str]:
        """Detect conflicts with other goals"""
        conflicts = []
        
        # Check resource conflicts
        resource_conflicts = await self._check_resource_conflicts(goal)
        conflicts.extend(resource_conflicts)
        
        # Check temporal conflicts
        temporal_conflicts = await self._check_temporal_conflicts(goal)
        conflicts.extend(temporal_conflicts)
        
        # Check priority conflicts
        priority_conflicts = await self._check_priority_conflicts(goal)
        conflicts.extend(priority_conflicts)
        
        # Record conflicts
        if conflicts:
            self.active_conflicts[goal.goal_id] = set(conflicts)
            
        return conflicts
        
    async def resolve_conflict(self, goal_id: str, conflict_id: str) -> bool:
        """Try to resolve a conflict between goals"""
        if goal_id not in self.active_conflicts:
            return True
            
        if conflict_id not in self.active_conflicts[goal_id]:
            return True
            
        # Get conflict pattern
        pattern = await self._identify_conflict_pattern(goal_id, conflict_id)
        
        # Get resolution strategy
        strategy = self.resolution_strategies.get(pattern)
        if not strategy:
            return False
            
        # Try to resolve
        success = await strategy(goal_id, conflict_id)
        
        # Update conflict status
        if success:
            self.active_conflicts[goal_id].remove(conflict_id)
            if conflict_id in self.active_conflicts:
                self.active_conflicts[conflict_id].remove(goal_id)
                
        return success

    async def _check_resource_conflicts(self, goal: Goal) -> List[str]:
        """Check for resource conflicts with other goals"""
        conflicts = []
        plan = await self.planner.get_plan(goal.goal_id)
        if not plan:
            return conflicts
            
        for other_goal_id, other_alloc in self.resource_manager.allocated_resources.items():
            # Skip self
            if other_goal_id == goal.goal_id:
                continue
                
            # Check for overlapping resource requirements
            for resource, amount in plan.total_resources.items():
                if resource in other_alloc:
                    total_required = amount + other_alloc[resource]
                    if total_required > self.resource_manager.available_resources[resource]:
                        conflicts.append(other_goal_id)
                        break
                        
        return conflicts

    async def _check_temporal_conflicts(self, goal: Goal) -> List[str]:
        """Check for temporal conflicts with other goals"""
        conflicts = []
        execution_time = await self.scheduler._calculate_execution_time(goal)
        
        # Check each timeframe's schedule
        for timeframe, schedule in self.scheduler.schedule.items():
            for time, other_goal_id in schedule:
                if other_goal_id == goal.goal_id:
                    continue
                    
                # Check for temporal overlap
                if abs((time - execution_time).total_seconds()) < 3600:  # 1 hour overlap
                    conflicts.append(other_goal_id)
                    
        return conflicts

class GoalHierarchyManager:
    """Manages the hierarchical relationship between goals"""
    def __init__(self):
        self.hierarchy: Dict[str, Set[str]] = {}  # parent_id -> child_ids
        self.reverse_hierarchy: Dict[str, str] = {}  # child_id -> parent_id
        self.root_goals: Set[str] = set()
        
    def add_goal(self, goal_id: str, parent_id: Optional[str] = None) -> None:
        """Add a goal to the hierarchy"""
        if parent_id:
            if parent_id not in self.hierarchy:
                self.hierarchy[parent_id] = set()
            self.hierarchy[parent_id].add(goal_id)
            self.reverse_hierarchy[goal_id] = parent_id
        else:
            self.root_goals.add(goal_id)
            
    def remove_goal(self, goal_id: str) -> None:
        """Remove a goal from the hierarchy"""
        # Remove from parent
        if goal_id in self.reverse_hierarchy:
            parent_id = self.reverse_hierarchy[goal_id]
            self.hierarchy[parent_id].remove(goal_id)
            del self.reverse_hierarchy[goal_id]
        else:
            self.root_goals.discard(goal_id)
            
        # Remove children
        if goal_id in self.hierarchy:
            children = self.hierarchy[goal_id].copy()
            for child_id in children:
                self.remove_goal(child_id)
            del self.hierarchy[goal_id]
            
    def get_subtree(self, goal_id: str) -> Set[str]:
        """Get all goals in the subtree rooted at goal_id"""
        subtree = {goal_id}
        if goal_id in self.hierarchy:
            for child_id in self.hierarchy[goal_id]:
                subtree.update(self.get_subtree(child_id))
        return subtree
        
    def get_ancestors(self, goal_id: str) -> List[str]:
        """Get all ancestors of a goal"""
        ancestors = []
        current = goal_id
        while current in self.reverse_hierarchy:
            parent = self.reverse_hierarchy[current]
            ancestors.append(parent)
            current = parent
        return ancestors

class GoalAnalyzer:
    """Analyzes goal patterns and generates insights"""
    def __init__(self):
        self.success_patterns: Dict[str, List[dict]] = {}
        self.failure_patterns: Dict[str, List[dict]] = {}
        self.completion_times: Dict[GoalType, List[float]] = {}
        self.success_rates: Dict[GoalType, float] = {}
        
    async def analyze_goal_completion(self, goal: Goal) -> dict:
        """Analyze a completed goal"""
        # Record completion metrics
        completion_time = (datetime.now() - goal.creation_time).total_seconds()
        if goal.type not in self.completion_times:
            self.completion_times[goal.type] = []
        self.completion_times[goal.type].append(completion_time)
        
        # Update success rates
        if goal.type not in self.success_rates:
            self.success_rates[goal.type] = 0
        total_goals = len(self.completion_times[goal.type])
        success_count = sum(1 for g in self.goals.values() 
                          if g.type == goal.type and g.status == GoalStatus.COMPLETED)
        self.success_rates[goal.type] = success_count / total_goals
        
        # Extract patterns
        if goal.status == GoalStatus.COMPLETED:
            pattern = await self._extract_success_pattern(goal)
            if goal.type not in self.success_patterns:
                self.success_patterns[goal.type] = []
            self.success_patterns[goal.type].append(pattern)
        else:
            pattern = await self._extract_failure_pattern(goal)
            if goal.type not in self.failure_patterns:
                self.failure_patterns[goal.type] = []
            self.failure_patterns[goal.type].append(pattern)
            
        return {
            'completion_time': completion_time,
            'success_rate': self.success_rates[goal.type],
            'pattern': pattern
        }
        
    async def generate_insights(self) -> dict:
        """Generate insights about goal patterns"""
        insights = {
            'completion_times': {
                goal_type: {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times)
                }
                for goal_type, times in self.completion_times.items()
            },
            'success_rates': self.success_rates,
            'common_patterns': {
                'success': self._find_common_patterns(self.success_patterns),
                'failure': self._find_common_patterns(self.failure_patterns)
            },
            'recommendations': await self._generate_recommendations()
        }
        return insights
