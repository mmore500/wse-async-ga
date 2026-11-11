"""Validators for rtc_p2pTarget values in a runtime config.

Takes comptime configs json as argument and runtime config npz on stdin.
"""

import io
import json
import sys

import numpy as np


def validate_rtc_p2pTarget_in_bounds(
    data: np.ndarray, comptime_config: dict
) -> None:
    has_target = data != 0xFFFFFFFF
    col, row = data >> 16, data & 0xFFFF
    assert (col[has_target] < comptime_config["nCol"]).all(), "col off grid"
    assert (row[has_target] < comptime_config["nRow"]).all(), "row off grid"


if __name__ == "__main__":
    comptime_config = json.loads(sys.argv[1])
    data = np.load(io.BytesIO(sys.stdin.buffer.read()))["rtc_p2pTarget"]
    print(f"validating rtc_p2pTarget {data.shape=}")
    validate_rtc_p2pTarget_in_bounds(data, comptime_config)
