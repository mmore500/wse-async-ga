"""Validators for rtc_globalSeed values in a runtime config.

Takes comptime configs json as argument and runtime config npz on stdin.
"""

import io
import json
import sys

import numpy as np


def validate_rtc_globalSeed_uniform(data: np.ndarray, comptime: dict) -> None:
    assert len(np.unique(data)) == 1, "seed differs between PEs"


def validate_rtc_globalSeed_local_seeds_fit_u32(
    data: np.ndarray, comptime: dict
) -> None:
    # kernel computes local seeds as globalSeed * nRow * nCol + whoami + 1
    assert (int(data.max()) + 1) * data.size < 2**32, "local seeds overflow"


if __name__ == "__main__":
    comptime = json.loads(sys.argv[1])
    data = np.load(io.BytesIO(sys.stdin.buffer.read()))["rtc_globalSeed"]
    print(f"validating rtc_globalSeed {data.shape=}")
    validate_rtc_globalSeed_uniform(data, comptime)
    validate_rtc_globalSeed_local_seeds_fit_u32(data, comptime)
