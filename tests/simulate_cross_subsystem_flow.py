
import logging
from mind_integration_system import MindIntegration, ExampleCognitiveTask

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def simulate_flow():
    logging.info("Starting cross-subsystem flow simulation...")

    # Initialize the Mind Integration system
    mind = MindIntegration()

    # Step 1: Simulate sensory input
    sensory_data = {"input_type": "visual", "data": "A red ball"}
    logging.info(f"Sensory data collected: {sensory_data}")
    preprocessed_data = mind.sensory.preprocess(sensory_data)
    logging.info(f"Preprocessed sensory data: {preprocessed_data}")

    # Step 2: Integrate context
    context_data = mind.context.integrate_context(preprocessed_data)
    logging.info(f"Context data integrated: {context_data}")

    # Step 3: Retrieve relevant memory
    relevant_memory = mind.memory.retrieve_relevant(context_data)
    logging.info(f"Relevant memory retrieved: {relevant_memory}")

    # Step 4: Reason with LLM
    decisions = mind.llm.reason(context_data, relevant_memory)
    logging.info(f"Decisions from LLM: {decisions}")

    # Step 5: Update goals based on decisions
    mind.goal_manager.update_goals(decisions)
    active_goals = mind.goal_manager.get_active_goals()
    logging.info(f"Active goals updated: {active_goals}")

    # Step 6: Add goals as tasks and execute them
    for goal in active_goals:
        mind.add_task(goal.priority, goal)
    logging.info(f"Tasks added to the queue: {mind.task_queue.queue}")
    mind.process_tasks()
    logging.info("All tasks processed.")

if __name__ == "__main__":
    simulate_flow()
