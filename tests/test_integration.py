
import unittest
from mind_integration_system import MindIntegration, ExampleCognitiveTask
from memory_subsystem import MemorySubsystem
from sensory_subsystem import SensorySubsystem
from goal_management import GoalManager

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.mind = MindIntegration()

    def test_sensory_to_context(self):
        # Simulate sensory data collection and preprocessing
        sensory_data = {"input_type": "visual", "data": "example visual input"}
        preprocessed_data = self.mind.sensory.preprocess(sensory_data)
        context_data = self.mind.context.integrate_context(preprocessed_data)
        self.assertIn("input_type", context_data)
        self.assertEqual(context_data["input_type"], "visual")

    def test_memory_retrieval(self):
        # Simulate memory retrieval based on context
        context_data = {"input_type": "visual", "data": "example context data"}
        relevant_memory = self.mind.memory.retrieve_relevant(context_data)
        self.assertIsInstance(relevant_memory, list)  # Assuming memory retrieval returns a list

    def test_llm_reasoning(self):
        # Simulate LLM reasoning with context and memory
        context_data = {"context": "example context"}
        memory_data = ["example memory item"]
        decisions = self.mind.llm.reason(context_data, memory_data)
        self.assertIsInstance(decisions, list)
        self.assertTrue(len(decisions) > 0)

    def test_goal_updates(self):
        # Simulate goal updates based on LLM decisions
        decisions = [{"type": "goal", "priority": 1, "description": "Test Goal"}]
        self.mind.goal_manager.update_goals(decisions)
        active_goals = self.mind.goal_manager.get_active_goals()
        self.assertTrue(len(active_goals) > 0)
        self.assertEqual(active_goals[0].description, "Test Goal")

    def test_task_execution(self):
        # Simulate task execution by adding a cognitive task
        task = ExampleCognitiveTask()
        self.mind.add_task(1, task)
        self.assertFalse(self.mind.task_queue.empty())
        # Process the task and ensure the queue is empty afterward
        self.mind.process_tasks()
        self.assertTrue(self.mind.task_queue.empty())

if __name__ == "__main__":
    unittest.main()
