import click
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from typing import Tuple


def load_data(load_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(os.path.join(load_path, 'data.csv'))
    target = pd.read_csv(os.path.join(load_path, 'target.csv'))
    return data, target


def save_data(save_to: str,
              x_train: pd.DataFrame,
              y_train: pd.Series,
              x_test: pd.DataFrame,
              y_test: pd.Series) -> None:
    os.makedirs(save_to, exist_ok=True)
    x_train.to_csv(os.path.join(save_to, 'data_train.csv'), index=False)
    y_train.to_csv(os.path.join(save_to, 'target_train.csv'), index=False)
    x_test.to_csv(os.path.join(save_to, 'data_test.csv'), index=False)
    y_test.to_csv(os.path.join(save_to, 'target_test.csv'), index=False)


@click.command()
@click.option('--data_path', required=True)
@click.option('--save_to', required=True)
@click.option('--test_size', default=0.25)
@click.option('--random_seed', default=42)
def split_data(data_path: str, save_to: str, test_size: float, random_seed: int) -> None:
    features, target = load_data(data_path)
    features_train, features_test, targets_train, targets_test = train_test_split(features,
                                                                                  target,
                                                                                  train_size=test_size,
                                                                                  random_state=random_seed)
    save_data(save_to, features_train, targets_train, features_test, targets_test)


if __name__ == '__main__':
    split_data()
