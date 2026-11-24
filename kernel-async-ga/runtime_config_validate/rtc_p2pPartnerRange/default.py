"""Validators for rtc_p2pPartnerRange values in a runtime config.

Takes comptime configs json as argument and runtime config npz on stdin.
"""

import io
import json
import sys

import numpy as np


def validate_rtc_p2pPartnerRange_in_bounds(
    data: np.ndarray, target: np.ndarray, comptime_config: dict
) -> None:
    range_col, range_row = data >> 16, data & 0xFFFF
    assert (range_col < comptime_config["nCol"]).all(), "col range off grid"
    assert (range_row < comptime_config["nRow"]).all(), "row range off grid"
    if target is None:  # no p2p targets to offset
        return
    has_target = target != 0xFFFFFFFF
    col, row = target >> 16, target & 0xFFFF
    far_col, far_row = col + range_col, row + range_row
    assert (
        far_col[has_target] < comptime_config["nCol"]
    ).all(), "col past range off grid"
    assert (
        far_row[has_target] < comptime_config["nRow"]
    ).all(), "row past range off grid"


if __name__ == "__main__":
    comptime_config = json.loads(sys.argv[1])
    runtime_config = np.load(io.BytesIO(sys.stdin.buffer.read()))
    data = runtime_config["rtc_p2pPartnerRange"]
    target = (
        runtime_config["rtc_p2pTarget"]
        if "rtc_p2pTarget" in runtime_config
        else None
    )
    print(f"validating rtc_p2pPartnerRange {data.shape=}")
    validate_rtc_p2pPartnerRange_in_bounds(data, target, comptime_config)
