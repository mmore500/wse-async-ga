"""Validators for rtc_nRowSubgrid values in a runtime config.

Takes comptime configs json as argument and runtime config npz on stdin.
"""

import io
import json
import sys

import numpy as np


def validate_rtc_nRowSubgrid_boundaries(
    data: np.ndarray, comptime_config: dict
) -> None:
    """Check that vertically adjacent PEs agree on whether connected."""
    size = data.squeeze(axis=2).astype(np.int64)  # errors unless 1 deep
    y = np.indices(size.shape)[0]
    first = (size != 0) & (y % np.maximum(size, 1) == 0)  # 0: no subgrids
    last = (size != 0) & (y % np.maximum(size, 1) == size - 1)
    num_bad = (last[:-1, :] != first[1:, :]).sum()
    assert not num_bad, f"{num_bad} PE pairs disagree on subgrid boundary"


if __name__ == "__main__":
    comptime_config = json.loads(sys.argv[1])
    data = np.load(io.BytesIO(sys.stdin.buffer.read()))["rtc_nRowSubgrid"]
    print(f"validating rtc_nRowSubgrid {data.shape=}")
    validate_rtc_nRowSubgrid_boundaries(data, comptime_config)
