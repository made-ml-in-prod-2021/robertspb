import click
import os
import pandas as pd
import pickle
import yaml

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from typing import Tuple


def load_data(load_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(os.path.join(load_path, 'data_test.csv'))
    target = pd.read_csv(os.path.join(load_path, 'target_test.csv'))
    return data, target


def load_model(model_path: str) -> LogisticRegression:
    with open(os.path.join(model_path, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    return model


def save_report(save_to: str, report: str) -> None:
    with open(os.path.join(save_to, "validation_report.yaml"), "w") as f:
        f.write(yaml.dump(report))


@click.command()
@click.option("--data_path", required=True)
@click.option("--model_path", required=True)
def validate(data_path: str, model_path: str):
    data_test, target_test = load_data(data_path)
    model = load_model(model_path)
    predicted = model.predict(data_test)
    report = classification_report(target_test, predicted)
    save_report(model_path, report)


if __name__ == "__main__":
    validate()
