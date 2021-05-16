from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class SplitConfig:
    name: str = MISSING
    test_size: float = MISSING

