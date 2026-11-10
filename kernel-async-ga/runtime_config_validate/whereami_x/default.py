"""Checks data for whereami_x, in runtime config npz file from stdin."""

import io
import sys

import numpy as np

if __name__ == "__main__":
    data = np.load(io.BytesIO(sys.stdin.buffer.read()))["whereami_x"]
    print(f"validating whereami_x {data.shape=}")
    assert data.max() < data.shape[1], "PE columns outside grid"
