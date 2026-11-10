import numpy as np


def validate(data: np.ndarray, params: dict) -> str:
    if len(np.unique(data)) != 1:
        return "seed differs between PEs"
    # kernel computes local seeds as globalSeed * nRow * nCol + whoami + 1
    if (int(data.max()) + 1) * params["nRow"] * params["nCol"] >= 2**32:
        return "seed too large, local seeds would overflow u32"
    return ""
