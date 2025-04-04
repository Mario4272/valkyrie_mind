from typing import Dict

class ValueSystem:
    def __init__(self):
        self.values: Dict[str, float] = {
            "truth": 0.9,
            "safety": 1.0,
            "autonomy": 0.8,
            "growth": 0.85
        }

    def update_value(self, key: str, weight: float) -> None:
        self.values[key] = weight

    def evaluate_action(self, action_traits: Dict[str, float]) -> float:
        # Dot product-style value alignment
        score = 0.0
        for k, v in action_traits.items():
            score += self.values.get(k, 0.0) * v
        return score
