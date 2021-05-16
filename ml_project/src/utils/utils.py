import hydra
import logging
import os
from pathlib import Path


logger = logging.getLogger('ml')


def get_absolute_path(path):
    logger.info(f'Creating absolute path from: {path}')
    base_path = Path(hydra.utils.get_original_cwd())
    path = os.path.normpath(path)
    absolute_path = os.path.join(base_path, path)
    logger.info(f'Created absolute path: {absolute_path}')
    return absolute_path
