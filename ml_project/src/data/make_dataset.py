import pandas as pd
import logging

from sklearn.model_selection import train_test_split
from typing import Tuple

from src.config import SplitConfig


logger = logging.getLogger('ml')


def read_data(data_path: str) -> pd.DataFrame:
    logger.info(f'Start reading data from path: {data_path}')
    data = pd.read_csv(data_path)
    logger.info(f'Finished reading data - success')
    return data


def edit_data(data: pd.DataFrame, target: str = 'target') -> pd.DataFrame:
    logger.info('Start one-hot encoding for categorical features')
    categorical_values = []
    for column in data.columns:
        if len(data[column].unique()) <= 10 and column != target:
            categorical_values.append(column)
    dataset = pd.get_dummies(data, columns=categorical_values)
    logger.info('Finished editing features in data - success')
    return dataset


def split_data_train_val(data: pd.DataFrame, split_params: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f'Start splitting dataset in proportion: '
                f'train - {1 - split_params.test_size}, test - {split_params.test_size}')
    train_data, val_data = train_test_split(data, test_size=split_params.test_size)
    logger.info('Finished splitting dataset - success')
    return train_data, val_data
