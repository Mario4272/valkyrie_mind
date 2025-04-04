from typing import Dict

class RelationshipMemory:
    def __init__(self):
        self.relationships: Dict[str, float] = {}  # e.g., {"Paige": 0.9}

    def update_relationship(self, name: str, affinity: float) -> None:
        self.relationships[name] = affinity

    def get_affinity(self, name: str) -> float:
        return self.relationships.get(name, 0.5)  # Neutral by default
