import pandas as pd
import logging

from typing import Tuple


logger = logging.getLogger('ml')


def get_x_y(data: pd.DataFrame, target_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f'Start separating target "{target_name}" from features')
    target = data[target_name]
    features = data.drop(target_name, axis=1)
    logger.info('Finished separating - success')
    return features, target
