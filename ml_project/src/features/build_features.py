import pandas as pd
import logging

from typing import Tuple


# logger = logging.getLogger()


def get_x_y(data: pd.DataFrame, target_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # logger.info('')
    target = data[target_name]
    features = data.drop(target_name, axis=1)
    # logger.info('')
    return features, target
