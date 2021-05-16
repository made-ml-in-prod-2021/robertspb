import hydra
import logging.config
import pickle
import yaml

from sklearn.metrics import classification_report
from typing import Any

from src.utils import get_absolute_path
from src.data import read_data
from src.data import edit_data
from src.data import split_data_train_val
from src.features import get_x_y
from src.features import CustomTransformer
from .predict_model import predict
from src.config import (SaveModelConfig,
                        SaveReportConfig,
                        Config)


logging.config.fileConfig('configs/logging.conf')
logger = logging.getLogger('ml')


def save_model(model: Any, cfg: SaveModelConfig) -> None:
    logger.info(f'Start saving model to directory: {cfg.model_out_path} with name: {cfg.model_filename}')
    out_path = cfg.model_out_path + cfg.model_filename
    out_path = get_absolute_path(out_path)
    with open(out_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f'Model saved')


def save_report(metrics: dict, cfg: SaveReportConfig) -> None:
    logger.info(f'Start saving report to directory: {cfg.report_out_path} with name: {cfg.report_filename}')
    out_path = cfg.report_out_path + cfg.report_filename
    out_path = get_absolute_path(out_path)
    with open(out_path, 'w') as f:
        f.write(yaml.dump(metrics))
    logger.info(f'Report created and saved')


def start_pipeline(cfg: Config):
    logger.info(f'Start model training pipeline')

    logger.info('Loading data from path')
    data_path = get_absolute_path(cfg.main.data_path)
    data = read_data(data_path)
    logger.debug(f'Data shape: {data.shape}')

    data = edit_data(data, cfg.main.target_name)
    logger.debug(f'Data shape after one-hot encoding: {data.shape}')

    train_data, val_data = split_data_train_val(data, cfg.split)

    logger.info('Getting features and target for train and validation datasets')
    train_x, train_y = get_x_y(train_data, cfg.main.target_name)
    val_x, val_y = get_x_y(val_data, cfg.main.target_name)
    logger.debug(f'train X shape: {train_x.shape}, train y shape: {train_y.shape}'
                 f'validation X shape: {val_x.shape}, validation y shape: {val_y.shape}')

    logger.info('Starting transforming data with StandardScaler')
    transformer = CustomTransformer()
    transformer.fit(train_x)
    train_x = transformer.transform(train_x)
    val_x = transformer.transform(val_x)
    logger.info('Finished transformation')

    logger.info(f'Importing model config: {cfg.model}')
    model = hydra.utils.instantiate(cfg.model)
    logger.info('Training model on train dataset')
    model.fit(train_x, train_y)

    logger.info('Getting predictions for validation dataset')
    predictions = predict(model, val_x)
    logger.info('Start generating report')
    metrics = classification_report(val_y, predictions, output_dict=True)

    logger.info('Saving model and report')
    logger.debug(f'Saving configs for model: {cfg.main.out_model} \n'
                 f'for report: {cfg.main.reports}')
    save_report(metrics, cfg.main.reports)
    save_model(model, cfg.main.out_model)

    logger.info('Finished evaluating pipeline')


@hydra.main(config_name="config", config_path="../../configs")
def start(cfg):
    start_pipeline(cfg)


if __name__ == "__main__":
    start()
