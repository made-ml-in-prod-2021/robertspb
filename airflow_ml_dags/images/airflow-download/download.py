import os

import click
import numpy as np
import pandas as pd

from typing import Tuple


def get_sample_dataset(size: int,
                       random_seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(random_seed)
    data = pd.DataFrame()
    data['age'] = np.random.normal(loc=54, scale=9, size=size).astype(int)
    data['sex'] = np.random.binomial(n=1, p=0.7, size=size).astype(int)
    data['cp'] = np.random.randint(low=0, high=4, size=size).astype(int)
    data['trestbps'] = np.random.normal(loc=131, scale=18, size=size).astype(int)
    data['chol'] = np.random.normal(loc=246, scale=51, size=size).astype(int)
    data['fbs'] = np.random.binomial(n=1, p=0.15, size=size).astype(int)
    data['restecg'] = np.random.randint(low=0, high=3, size=size).astype(int)
    data['thalach'] = np.random.normal(loc=150, scale=23, size=size).astype(int)
    data['exang'] = np.random.binomial(n=1, p=0.33, size=size).astype(int)
    data['oldpeak'] = np.clip(np.random.normal(loc=1, scale=2, size=size), 0, None).astype(int)
    data['slope'] = np.random.randint(low=0, high=3, size=size).astype(int)
    data['ca'] = np.random.randint(low=0, high=5, size=size).astype(int)
    data['thal'] = np.random.randint(low=0, high=4, size=size).astype(int)
    data['target'] = np.random.binomial(n=1, p=0.55, size=size).astype(int)
    target = data['target']
    data = data.drop('target', axis=1)
    return data, target


@click.command()
@click.option('--output_dir', required=True)
@click.option('--size', default=100)
@click.option('--random_seed', default=42)
def load_data(output_dir: str, size: int, random_seed: int) -> None:
    data, target = get_sample_dataset(size=size, random_seed=random_seed)

    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, 'data.csv'), index=False)
    target.to_csv(os.path.join(output_dir, 'target.csv'), index=False)


if __name__ == '__main__':
    load_data()
