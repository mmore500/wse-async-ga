"""Checks data for nRowSubgrid, in runtime config npz file from stdin."""

import io
import sys

import numpy as np

if __name__ == "__main__":
    data = np.load(io.BytesIO(sys.stdin.buffer.read()))["nRowSubgrid"]
    print(f"validating nRowSubgrid {data.shape=}")
    assert len(np.unique(data)) == 1, "subgrid size differs between PEs"
