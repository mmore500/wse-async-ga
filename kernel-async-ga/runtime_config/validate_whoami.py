import numpy as np


def validate(data: np.ndarray, params: dict) -> str:
    if len(np.unique(data)) != data.size:
        return "PE ids are not unique"
    return ""
