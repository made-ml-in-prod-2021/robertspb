import click
import logging
import pandas as pd
import pickle

from src.data import read_data
from src.data import edit_data


logger = logging.getLogger('ml')


def predict(model: dict, data: pd.DataFrame) -> pd.Series:
    logger.info(f'Start predicting')
    logger.debug(f'Data shape for predicting: {data.shape}')
    predicted_data = model.predict(data)
    logger.info('Finished predicting - success')
    return predicted_data


def load_model(path: str) -> dict:
    logger.info(f'Loading model from path: {path}')
    with open(path, 'rb') as f:
        model = pickle.load(f)
    logger.info('Model loaded')
    return model


@click.command()
@click.option('--model_path', default='models/model.pkl')
@click.option('--data_path', default='data/raw/heart.csv')
@click.option('--out_path', default='predictions/predictions.csv')
def cmd_predict(model_path: str, data_path: str, out_path: str) -> None:
    model = load_model(model_path)
    data = read_data(data_path)
    data = edit_data(data)
    if 'target' in data.columns:
        data = data.drop('target', axis=1)
    predictions = predict(model, data)
    pd.DataFrame(predictions).to_csv(out_path)


if __name__ == '__main__':
    cmd_predict()
