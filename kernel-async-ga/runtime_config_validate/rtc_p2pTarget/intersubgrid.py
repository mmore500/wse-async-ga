"""Validator that rtc_p2pTarget values in a runtime config stay in subgrids.

Takes comptime configs json as argument and runtime config npz on stdin.
"""

import io
import json
import sys

import numpy as np


def validate_rtc_p2pTarget_within_subgrid(
    data: np.ndarray, nRowSubgrid: np.ndarray, nColSubgrid: np.ndarray
) -> None:
    """Check p2p targets share a subgrid, given per-PE subgrid sizes."""
    nRow, nCol = data.shape[:2]
    rows, cols = np.indices((nRow, nCol))
    nRowSubgrid = np.broadcast_to(nRowSubgrid, (nRow, nCol)).astype(np.int64)
    nColSubgrid = np.broadcast_to(nColSubgrid, (nRow, nCol)).astype(np.int64)
    # subgrids identified by their top-left PE's row and col
    subgridTopRow = rows - rows % np.where(nRowSubgrid, nRowSubgrid, nRow)
    subgridLeftCol = cols - cols % np.where(nColSubgrid, nColSubgrid, nCol)
    target = data.squeeze(axis=2)  # enforce depth exactly 1
    has = target != 0xFFFFFFFF
    col, row = target[has] >> 16, target[has] & 0xFFFF
    is_intersubgrid = (subgridTopRow[row, col] != subgridTopRow[has]) | (
        subgridLeftCol[row, col] != subgridLeftCol[has]
    )
    num_bad = is_intersubgrid.sum()
    assert not num_bad, f"{num_bad} PEs have target in another subgrid"


if __name__ == "__main__":
    comptime_config = json.loads(sys.argv[1])
    runtime_config = np.load(io.BytesIO(sys.stdin.buffer.read()))
    data = runtime_config["rtc_p2pTarget"]
    print(f"validating rtc_p2pTarget {data.shape=}")

    # subgrid sizes are runtime config too, defaulting to comptime_config
    nRowSubgrid, nColSubgrid = (
        (
            runtime_config[f"rtc_{key}"].squeeze(axis=2)
            if f"rtc_{key}" in runtime_config
            else comptime_config[key]
        )
        for key in ("nRowSubgrid", "nColSubgrid")
    )
    print(f"{np.unique(nRowSubgrid)=} {np.unique(nColSubgrid)=}")
    validate_rtc_p2pTarget_within_subgrid(data, nRowSubgrid, nColSubgrid)
