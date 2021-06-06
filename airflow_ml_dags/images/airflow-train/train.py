import click
import os
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from typing import Tuple


def load_data(load_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(os.path.join(load_path, 'data_train.csv'))
    target = pd.read_csv(os.path.join(load_path, 'target_train.csv'))
    return data, target


def save_model(save_to: str, model: LogisticRegression) -> None:
    os.makedirs(save_to, exist_ok=True)
    with open(os.path.join(save_to, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)


@click.command()
@click.option('--data_path', required=True)
@click.option('--save_to', required=True)
@click.option('--random_seed', default=42)
def train_model(data_path: str, save_to: str, random_seed: int) -> None:
    data_train, target_train = load_data(data_path)
    model = LogisticRegression(random_state=random_seed)
    model.fit(data_train, target_train)
    save_model(save_to, model)


if __name__ == '__main__':
    train_model()
