"""Validators for rtc_p2pSpread values in a runtime config.

Takes comptime configs json as argument and runtime config npz on stdin.
"""

import io
import json
import sys

import numpy as np


def validate_rtc_p2pSpread_in_bounds(
    data: np.ndarray, target: np.ndarray, comptime_config: dict
) -> None:
    spread_col, spread_row = data >> 16, data & 0xFFFF
    assert (spread_col < comptime_config["nCol"]).all(), "col spread off grid"
    assert (spread_row < comptime_config["nRow"]).all(), "row spread off grid"
    if target is None:  # no p2p targets to offset
        return
    has_target = target != 0xFFFFFFFF
    col, row = target >> 16, target & 0xFFFF
    far_col, far_row = col + spread_col, row + spread_row
    assert (
        far_col[has_target] < comptime_config["nCol"]
    ).all(), "col past spread off grid"
    assert (
        far_row[has_target] < comptime_config["nRow"]
    ).all(), "row past spread off grid"


if __name__ == "__main__":
    comptime_config = json.loads(sys.argv[1])
    runtime_config = np.load(io.BytesIO(sys.stdin.buffer.read()))
    data = runtime_config["rtc_p2pSpread"]
    target = None  # no p2p targets to offset
    if "rtc_p2pTarget" in runtime_config:
        target = runtime_config["rtc_p2pTarget"]
    print(f"validating rtc_p2pSpread {data.shape=}")
    validate_rtc_p2pSpread_in_bounds(data, target, comptime_config)
