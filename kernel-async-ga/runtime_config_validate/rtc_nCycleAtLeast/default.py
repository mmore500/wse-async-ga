"""Validators for rtc_nCycleAtLeast values in a runtime config.

Takes comptime configs json as argument and runtime config npz on stdin.
"""

import io
import json
import sys

import numpy as np


def validate_rtc_nCycleAtLeast_uniform(
    data: np.ndarray, comptime: dict
) -> None:
    assert len(np.unique(data)) == 1, "cycle limit differs between PEs"


if __name__ == "__main__":
    comptime = json.loads(sys.argv[1])
    data = np.load(io.BytesIO(sys.stdin.buffer.read()))["rtc_nCycleAtLeast"]
    print(f"validating rtc_nCycleAtLeast {data.shape=}")
    validate_rtc_nCycleAtLeast_uniform(data, comptime)
