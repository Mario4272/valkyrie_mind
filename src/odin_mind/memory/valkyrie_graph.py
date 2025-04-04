# valkyrie_graph/valkyrie_graph.py
# """ ValkyrieGraph: Symbolic-perceptual memory engine based on the P-Graph architecture. 
# Supports emotionally weighted, context-aware memory traversal and storage."""

import pickle
from typing import Dict, Optional, List
from odin_mind.memory.perceptual_frame import PerceptualFrame
from datetime import datetime

class ValkyrieGraph:
    def __init__(self):
        self.frames: Dict[str, PerceptualFrame] = {}

    def add_frame(self, frame: PerceptualFrame):
        self.frames[frame.id] = frame

    def get_frame(self, frame_id: str) -> Optional[PerceptualFrame]:
        return self.frames.get(frame_id)

    def relate_frames(self, src_id: str, dest_id: str, relation: str):
        src_frame = self.get_frame(src_id)
        if src_frame:
            src_frame.add_relation(relation, dest_id)

    def traverse(self, start_id: str, relation: str) -> List[PerceptualFrame]:
        start = self.get_frame(start_id)
        if not start:
            return []
        related_ids = start.related_frames.get(relation, [])
        return [self.get_frame(fid) for fid in related_ids if fid in self.frames]
    
    def access_frame(self, frame_id: str) -> Optional[PerceptualFrame]:    
        frame = self.get_frame(frame_id)
        if frame:
            frame.salience["frequency"] += 1
            frame.salience["last_accessed"] = datetime.utcnow().timestamp()
        return frame
    
    def filter_by_emotion(self, emotion: str, threshold: float = 0.5) -> List[PerceptualFrame]:
        # Get all frames where a specific emotion crosses the threshold
        return [
            frame for frame in self.frames.values()
            if frame.emotion_vector and frame.emotion_vector.get(emotion, 0) >= threshold
        ]

    def get_top_salient_frames(self, top_n: int = 5) -> List[PerceptualFrame]:
        # Return the N most 'salient' (important) frames by emotion + recall frequency
        return sorted(
            self.frames.values(),
            key=lambda f: (
                f.salience["emotional_weight"] +
                f.salience["frequency"]
            ),
            reverse=True
        )[:top_n]

    def save_graph(self, filepath: str):
        # Save the entire memory graph to disk as a binary file
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_graph(filepath: str) -> "ValkyrieGraph":
        # Load a memory graph from disk (returns a new ValkyrieGraph instance)
        with open(filepath, "rb") as f:
            return pickle.load(f)
        
    def reflect_daily(self, top_n: int = 3) -> str:
        # Pull top N most salient frames and generate a text summary
        top_frames = self.get_top_salient_frames(top_n)

        if not top_frames:
            return "No significant experiences to reflect on today."

        summary = ["Today's most meaningful experiences:\n"]

        for idx, frame in enumerate(top_frames, 1):
            intent = frame.intent or "No specific intent"
            emotions = ", ".join(f"{k} ({v:.2f})" for k, v in frame.emotion_vector.items()) if frame.emotion_vector else "None"
            tags = ", ".join(frame.semantic_tags) if frame.semantic_tags else "No tags"

            summary.append(f"{idx}. Intent: {intent}")
            summary.append(f"   Emotions: {emotions}")
            summary.append(f"   Tags: {tags}\n")

        return "\n".join(summary)

    # Emotional Stablizer - Synthetic Coping Mechanism (SCM)
    def stabilize_emotion(self, target_emotion: str, threshold: float = 0.5, top_n: int = 3) -> List[PerceptualFrame]:
        """Retrieve up to N memories with a dominant expression of the target emotion.
        Useful for synthetic mood regulation."""
        matching_frames = self.filter_by_emotion(target_emotion, threshold)

        # Sort matching frames by emotional intensity for the target emotion
        sorted_frames = sorted(
            matching_frames,
            key=lambda f: f.emotion_vector.get(target_emotion, 0),
            reverse=True
        )

        return sorted_frames[:top_n]