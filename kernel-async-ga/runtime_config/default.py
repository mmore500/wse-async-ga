"""Runtime config that leaves all symbols at their defaults."""

import numpy as np


def generate(params: dict) -> "dict[str, np.ndarray]":
    """Return data to override symbols' defaults, keyed by symbol.

    Data should have shape (nRow, nCol, width); for example,

        shape = (params["nRow"], params["nCol"], 1)
        return {"globalSeed": np.full(shape, 42, dtype=np.uint32)}
    """
    return {}
