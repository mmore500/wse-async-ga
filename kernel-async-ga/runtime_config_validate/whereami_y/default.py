"""Checks data for whereami_y, in runtime config npz file from stdin."""

import io
import sys

import numpy as np

if __name__ == "__main__":
    data = np.load(io.BytesIO(sys.stdin.buffer.read()))["whereami_y"]
    print(f"validating whereami_y {data.shape=}")
    assert data.max() < data.shape[0], "PE rows outside grid"
