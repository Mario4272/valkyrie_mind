from typing import Dict
from datetime import datetime

class EmotionManager:
    def __init__(self):
        self.current_emotions: Dict[str, float] = {}

    def update_emotion(self, emotion: str, intensity: float) -> None:
        self.current_emotions[emotion] = intensity

    def get_dominant_emotion(self) -> str:
        if not self.current_emotions:
            return "neutral"
        return max(self.current_emotions.items(), key=lambda x: x[1])[0]

    def decay_emotions(self, decay_rate: float = 0.01) -> None:
        for key in list(self.current_emotions.keys()):
            self.current_emotions[key] *= (1 - decay_rate)
            if self.current_emotions[key] < 0.01:
                del self.current_emotions[key]
