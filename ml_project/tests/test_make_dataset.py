import os.path

import pandas as pd
import pytest

from src.data import read_data, split_data_train_val
from src.config import SplitConfig


@pytest.fixture
def data_path() -> str:
    return 'data/raw/heart.csv'


@pytest.fixture
def target_name() -> str:
    return 'target'


@pytest.fixture
def dataset(data_path) -> pd.DataFrame:
    data_path = os.path.abspath(data_path)
    return read_data(data_path)


@pytest.fixture()
def split_config() -> SplitConfig:
    return SplitConfig(test_size=0.3)


def test_read_data(data_path: str, target_name: str):
    data_path = os.path.abspath(data_path)
    data = read_data(data_path)
    assert len(data) > 50
    assert target_name in data.columns


def test_split_train_val_data(dataset: pd.DataFrame, split_config: SplitConfig):
    train_data, val_data = split_data_train_val(dataset, split_config)
    assert len(train_data) > 0
    assert len(val_data) > 0
