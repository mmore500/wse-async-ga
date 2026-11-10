import numpy as np


def validate(data: np.ndarray, params: dict) -> str:
    if data.max() >= params["nRow"]:
        return "PE coordinates outside grid"
    return ""
