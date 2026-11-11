"""Validators for rtc_nColSubgrid values in a runtime config.

Takes comptime configs json as argument and runtime config npz on stdin.
"""

import io
import json
import sys

import numpy as np


def validate_rtc_nColSubgrid_uniform(
    data: np.ndarray, comptime_config: dict
) -> None:
    assert len(np.unique(data)) == 1, "subgrid size differs between PEs"


if __name__ == "__main__":
    comptime_config = json.loads(sys.argv[1])
    data = np.load(io.BytesIO(sys.stdin.buffer.read()))["rtc_nColSubgrid"]
    print(f"validating rtc_nColSubgrid {data.shape=}")
    validate_rtc_nColSubgrid_uniform(data, comptime_config)
