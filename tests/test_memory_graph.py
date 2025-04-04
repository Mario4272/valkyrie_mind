# examples/test_graph.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from valkyrie_mind.memory.perceptual_frame import PerceptualFrame
from valkyrie_mind.memory.valkyrie_graph import ValkyrieGraph

# Create the graph
graph = ValkyrieGraph()

# Create frames
frame1 = PerceptualFrame(
    semantic_tags=["truck", "steering wheel", "pickup"],
    emotion_vector={"anticipation": 0.8},
    intent="Pick up wife from mall"
)

frame2 = PerceptualFrame(
    semantic_tags=["Starbucks", "coffee"],
    emotion_vector={"comfort": 0.6, "joy": 0.3},
    intent="Drive to Starbucks"
)

frame3 = PerceptualFrame(
    semantic_tags=["home", "relax"],
    emotion_vector={"calm": 0.9},
    intent="Arrive home and relax"
)

# Add to graph
graph.add_frame(frame1)
graph.add_frame(frame2)
graph.add_frame(frame3)

# Relate them
graph.relate_frames(frame1.id, frame2.id, "after")
graph.relate_frames(frame2.id, frame3.id, "after")
graph.relate_frames(frame3.id, frame1.id, "related")  # reflection loop

# Traverse forward from frame1
print("Forward timeline:")
for f in graph.traverse(frame1.id, "after"):
    print(f" - {f.intent} ({f.semantic_tags})")

# Traverse related frames from frame3
print("\nReflected memory from home:")
for f in graph.traverse(frame3.id, "related"):
    print(f" - {f.intent} ({f.semantic_tags})")

# Show Memories based on threshold
print("\nCalm memories (threshold 0.5):")
for f in graph.filter_by_emotion("calm", 0.5):
    print(f" - {f.intent} ({f.emotion_vector})")

# Show memories by saliency 
print("\nTop salient memories:")
for f in graph.get_top_salient_frames():
    print(f" - {f.intent} | Salience: {f.salience}")

# Save graph
graph.save_graph("memory_graph.vk")

# Load Graph
loaded_graph = ValkyrieGraph.load_graph("memory_graph.vk")

# Show Recalled Graph
print("\nLoaded Graph Recall:")
for f in loaded_graph.get_top_salient_frames():
    print(f" - {f.intent} | Salience: {f.salience}")

# Show Daily Reflections
print("\nDaily Reflection:")
print(graph.reflect_daily(top_n=3))

# Test SCM
print("\nEmotion Stabilization: Seeking calm")
for f in graph.stabilize_emotion("calm", threshold=0.4):
    print(f" - {f.intent} ({f.emotion_vector})")