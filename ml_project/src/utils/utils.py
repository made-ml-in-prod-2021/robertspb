import hydra
import os
from pathlib import Path


def get_absolute_path(path):
    base_path = Path(hydra.utils.get_original_cwd()).parent.parent
    path = os.path.normpath(path)
    return os.path.join(base_path, path)
