import numpy as np


def validate(data: np.ndarray, params: dict) -> str:
    if len(np.unique(data)) != 1:
        return "subgrid size differs between PEs"
    return ""
