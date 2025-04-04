from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Any
import asyncio
from datetime import datetime
import numpy as np
from abc import ABC, abstractmethod

from valkyrie_mind.valkyrie_mind.core.mind_integration_system import MindSystem

class ContextType(Enum):
    ENVIRONMENTAL = "environmental"  # Physical environment
    SITUATIONAL = "situational"      # Current situation
    TEMPORAL = "temporal"            # Time-related
    SOCIAL = "social"                # Social interactions