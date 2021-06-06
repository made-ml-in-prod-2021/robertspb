import click
import os
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def load_data(load_path: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(load_path, 'data.csv'))


def load_model(model_path: str) -> LogisticRegression:
    with open(os.path.join(model_path, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    return model


def load_transformer(model_path: str) -> StandardScaler:
    with open(os.path.join(model_path, "transformer.pkl"), "rb") as f:
        transformer = pickle.load(f)
    return transformer


def save_predictions(save_to: str, predictions: pd.Series) -> None:
    os.makedirs(save_to, exist_ok=True)
    predictions.to_csv(os.path.join(save_to, 'predictions.csv'), index=False)


@click.command()
@click.option('--data_path', required=True)
@click.option('--model_path', required=True)
@click.option('--save_to', required=True)
def train_model(data_path: str, model_path: str, save_to: str) -> None:
    data = load_data(data_path)
    model = load_model(model_path)
    transformer = load_transformer(model_path)
    transformed_data = transformer.transform(data)
    predictions = pd.Series(model.predict(transformed_data))
    save_predictions(save_to, predictions)


if __name__ == '__main__':
    train_model()
