"""Validators for rtc_p2pSendProbability values in a runtime config.

Takes comptime configs json as argument and runtime config npz on stdin.
"""

import io
import json
import sys

import numpy as np


def validate_rtc_p2pSendProbability_is_probability(
    data: np.ndarray, comptime_config: dict
) -> None:
    probability = data.view(np.float32)  # copied to device as raw bits
    assert np.isfinite(probability).all(), "send probability not finite"
    assert ((0 <= probability) & (probability <= 1)).all(), "not within 0 to 1"


def validate_rtc_p2pSendProbability_compiled_in(
    data: np.ndarray, comptime_config: dict
) -> None:
    # kernel only has p2p exchange if compiled with nonzero send
    # probability, otherwise runtime send probability has no effect
    probability = data.view(np.float32)
    assert (
        comptime_config["p2pSendProbability"] > 0 or (probability == 0).all()
    ), "nonzero runtime send probability, but p2p is not compiled in"


if __name__ == "__main__":
    comptime_config = json.loads(sys.argv[1])
    data = np.load(io.BytesIO(sys.stdin.buffer.read()))[
        "rtc_p2pSendProbability"
    ]
    print(f"validating rtc_p2pSendProbability {data.shape=}")
    validate_rtc_p2pSendProbability_is_probability(data, comptime_config)
    validate_rtc_p2pSendProbability_compiled_in(data, comptime_config)
