import numpy as np


def validate(data: np.ndarray, params: dict) -> str:
    if data.max() >= params["nCol"]:
        return "PE coordinates outside grid"
    return ""
