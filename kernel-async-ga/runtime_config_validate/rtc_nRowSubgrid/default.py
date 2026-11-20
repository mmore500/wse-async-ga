"""Validators for rtc_nRowSubgrid values in a runtime config.

Takes comptime configs json as argument and runtime config npz on stdin.
"""

import io
import json
import sys

import numpy as np


def validate_rtc_nRowSubgrid_walls(
    data: np.ndarray, comptime_config: dict
) -> None:
    """Check vertically adjacent PEs agree on whether they can migrate.

    Subgrid sizes may differ between PEs (e.g., nestedSubgridSizeFactor), but kernel.csl
    disables migration per PE, at y % n == 0 (N) and y % n == n - 1 (S).
    """
    size = data[:, :, 0].astype(np.int64)
    y = np.indices(size.shape)[0]
    first = (size != 0) & (y % np.maximum(size, 1) == 0)  # 0: no subgrids
    last = (size != 0) & (y % np.maximum(size, 1) == size - 1)
    num_bad = (last[:-1, :] != first[1:, :]).sum()
    assert not num_bad, f"{num_bad} PE pairs disagree on subgrid boundary"


if __name__ == "__main__":
    comptime_config = json.loads(sys.argv[1])
    data = np.load(io.BytesIO(sys.stdin.buffer.read()))["rtc_nRowSubgrid"]
    print(f"validating rtc_nRowSubgrid {data.shape=}")
    validate_rtc_nRowSubgrid_walls(data, comptime_config)
