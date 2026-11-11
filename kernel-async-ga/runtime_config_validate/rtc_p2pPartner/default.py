"""Validators for rtc_p2pPartner values in a runtime config.

Takes comptime configs json as argument and runtime config npz on stdin.
"""

import io
import json
import sys

import numpy as np


def validate_rtc_p2pPartner_in_bounds(
    data: np.ndarray, comptime_config: dict
) -> None:
    is_paired = data[:, :, 0] != 0xFFFFFFFF
    assert (data[:, :, 1] != 0xFFFFFFFF)[is_paired].all(), "partner lacks row"
    assert (
        data[:, :, 0][is_paired] < comptime_config["nCol"]
    ).all(), "col off grid"
    assert (
        data[:, :, 1][is_paired] < comptime_config["nRow"]
    ).all(), "row off grid"


if __name__ == "__main__":
    comptime_config = json.loads(sys.argv[1])
    data = np.load(io.BytesIO(sys.stdin.buffer.read()))["rtc_p2pPartner"]
    print(f"validating rtc_p2pPartner {data.shape=}")
    validate_rtc_p2pPartner_in_bounds(data, comptime_config)
