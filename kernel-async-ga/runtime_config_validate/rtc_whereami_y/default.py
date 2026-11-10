"""Validator for rtc_whereami_y values in a runtime config.

Takes comptime configs json as argument and runtime config npz on stdin.
"""

import io
import json
import sys

import numpy as np


def validate_rtc_whereami_y(data: np.ndarray, comptime: dict) -> None:
    assert data.max() < comptime["nRow"], "PE rows outside grid"


if __name__ == "__main__":
    comptime = json.loads(sys.argv[1])
    data = np.load(io.BytesIO(sys.stdin.buffer.read()))["rtc_whereami_y"]
    print(f"validating rtc_whereami_y {data.shape=}")
    validate_rtc_whereami_y(data, comptime)
