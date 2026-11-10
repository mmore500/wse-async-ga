"""Validator for rtc_nColSubgrid values in a runtime config.

Takes comptime configs json as argument and runtime config npz on stdin.
"""

import io
import json
import sys

import numpy as np


def validate_rtc_nColSubgrid(data: np.ndarray, comptime: dict) -> None:
    assert len(np.unique(data)) == 1, "subgrid size differs between PEs"


if __name__ == "__main__":
    comptime = json.loads(sys.argv[1])
    data = np.load(io.BytesIO(sys.stdin.buffer.read()))["rtc_nColSubgrid"]
    print(f"validating rtc_nColSubgrid {data.shape=}")
    validate_rtc_nColSubgrid(data, comptime)
