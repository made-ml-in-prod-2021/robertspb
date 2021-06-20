import click
import os
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from typing import Tuple


def load_data(load_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(os.path.join(load_path, 'data.csv'))
    target = pd.read_csv(os.path.join(load_path, 'target.csv'))
    return data, target


def save_data(save_to: str,
              data: pd.DataFrame,
              target: pd.Series,
              model_path: str,
              transformer: StandardScaler) -> None:
    os.makedirs(save_to, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    data.to_csv(os.path.join(save_to, 'data.csv'), index=False)
    target.to_csv(os.path.join(save_to, 'target.csv'), index=False)
    with open(os.path.join(model_path, 'transformer.pkl'), 'wb') as f:
        pickle.dump(transformer, f)


@click.command()
@click.option('--data_path', required=True)
@click.option('--save_to', required=True)
@click.option('--model_path', required=True)
def load_transform(data_path: str, save_to: str, model_path: str) -> None:
    data, target = load_data(data_path)

    transformer = StandardScaler()
    transformer.fit(data)
    transformed_data = pd.DataFrame(transformer.transform(data), columns=data.columns)

    save_data(save_to, transformed_data, target, model_path, transformer)


if __name__ == '__main__':
    load_transform()
