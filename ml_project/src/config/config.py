from dataclasses import dataclass
from omegaconf import MISSING
from typing import Any


@dataclass
class SplitConfig:
    name: str = MISSING
    test_size: float = MISSING


@dataclass
class SaveModelConfig:
    model_out_path: str = MISSING
    model_filename: str = MISSING


@dataclass
class SaveReportConfig:
    report_out_path: str = MISSING
    report_filename: str = MISSING


@dataclass
class MainConfig:
    data_path: str = MISSING
    target_name: str = MISSING
    out_model: SaveModelConfig = MISSING
    reports: SaveReportConfig = MISSING


@dataclass
class Config:
    main: MainConfig = MISSING
    split: SplitConfig = MISSING
    model: Any = MISSING
