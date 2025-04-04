from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Any
import asyncio
from datetime import datetime
import numpy as np
from abc import ABC, abstractmethod

class KnowledgeType(Enum):
    FACTUAL = "factual"           # Verified facts
    CONCEPTUAL = "conceptual"     # Abstract concepts
    PROCEDURAL = "procedural"     # How-to knowledge
    RELATIONAL = "relational"     # Relationships between entities
    TEMPORAL = "temporal"         # Time-based knowledge