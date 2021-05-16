import hydra
import logging
import pickle
import typing

import yaml
from sklearn.metrics import classification_report
from typing import Any

from src.utils import get_absolute_path
from src.data import read_data
from src.data import edit_data
from src.data import split_data_train_val
from src.features import get_x_y
from src.features import CustomTransformer
from predict_model import predict
from src.config import (SaveModelConfig,
                        SaveReportConfig,
                        Config)

# logger = logging.getLogger()


def save_model(model: Any, cfg: SaveModelConfig) -> None:
    out_path = cfg.model_out_path + cfg.model_filename
    out_path = get_absolute_path(out_path)
    with open(out_path, 'wb') as f:
        pickle.dump(model, f)


def save_report(metrics: dict, cfg: SaveReportConfig) -> None:
    out_path = cfg.report_out_path + cfg.report_filename
    out_path = get_absolute_path(out_path)
    with open(out_path, 'w') as f:
        f.write(yaml.dump(metrics))


def start_pipeline(cfg: Config):
    # logger.info()
    data_path = get_absolute_path(cfg.main.data_path)
    data = read_data(data_path)
    data = edit_data(data, cfg.main.target_name)
    train_data, val_data = split_data_train_val(data, cfg.split)
    train_x, train_y = get_x_y(train_data, cfg.main.target_name)
    val_x, val_y = get_x_y(val_data, cfg.main.target_name)
    transformer = CustomTransformer()
    transformer.fit(train_x)
    train_x = transformer.transform(train_x)
    val_x = transformer.transform(val_x)
    model = hydra.utils.instantiate(cfg.model)
    model.fit(train_x, train_y)
    predictions = predict(model, val_x)
    metrics = classification_report(val_y, predictions, output_dict=True)
    save_report(metrics, cfg.main.reports)
    save_model(model, cfg.main.out_model)


@hydra.main(config_name="config", config_path="../../configs")
def start(cfg):
    start_pipeline(cfg)


if __name__ == "__main__":
    start()
