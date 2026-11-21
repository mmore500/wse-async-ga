"""Validators for rtc_nColSubgrid values in a runtime config.

Takes comptime configs json as argument and runtime config npz on stdin.
"""

import io
import json
import sys

import numpy as np


def validate_rtc_nColSubgrid_boundaries(
    data: np.ndarray, comptime_config: dict
) -> None:
    """Check that horizontally adjacent PEs agree on whether connected."""
    size = data[:, :, 0].astype(np.int64)
    x = np.indices(size.shape)[1]
    first = (size != 0) & (x % np.maximum(size, 1) == 0)  # 0: no subgrids
    last = (size != 0) & (x % np.maximum(size, 1) == size - 1)
    num_bad = (last[:, :-1] != first[:, 1:]).sum()
    assert not num_bad, f"{num_bad} PE pairs disagree on subgrid boundary"


if __name__ == "__main__":
    comptime_config = json.loads(sys.argv[1])
    data = np.load(io.BytesIO(sys.stdin.buffer.read()))["rtc_nColSubgrid"]
    print(f"validating rtc_nColSubgrid {data.shape=}")
    validate_rtc_nColSubgrid_boundaries(data, comptime_config)
