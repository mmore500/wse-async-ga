"""Checks data for globalSeed, in runtime config npz file from stdin."""

import io
import sys

import numpy as np

if __name__ == "__main__":
    data = np.load(io.BytesIO(sys.stdin.buffer.read()))["globalSeed"]
    print(f"validating globalSeed {data.shape=}")
    assert len(np.unique(data)) == 1, "seed differs between PEs"
    # kernel computes local seeds as globalSeed * nRow * nCol + whoami + 1
    assert (int(data.max()) + 1) * data.size < 2**32, "local seeds overflow"
