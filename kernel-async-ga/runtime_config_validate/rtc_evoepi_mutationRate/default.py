"""Validators for rtc_evoepi_mutationRate values in a runtime config.

Takes comptime configs json as argument and runtime config npz on stdin.
"""

import io
import json
import sys

import numpy as np


def validate_rtc_evoepi_mutationRate(
    data: np.ndarray, comptime_config: dict
) -> None:
    rate = data.view(np.float32)
    assert np.isfinite(rate).all(), "mutation rate not finite"
    assert (rate >= 0.0).all(), "mutation rate negative"
    assert len(np.unique(rate)) == 1, "mutation rate differs between PEs"


if __name__ == "__main__":
    comptime_config = json.loads(sys.argv[1])
    data = np.load(io.BytesIO(sys.stdin.buffer.read()))[
        "rtc_evoepi_mutationRate"
    ]
    print(f"validating rtc_evoepi_mutationRate {data.shape=}")
    validate_rtc_evoepi_mutationRate(data, comptime_config)
