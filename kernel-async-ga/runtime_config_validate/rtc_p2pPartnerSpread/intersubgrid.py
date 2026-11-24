"""Validator that rtc_p2pPartnerSpread keeps p2p targets in their subgrid.

Takes comptime configs json as argument and runtime config npz on stdin.
"""

import io
import json
import sys

import numpy as np


def validate_rtc_p2pPartnerSpread_within_subgrid(
    data: np.ndarray,
    target: np.ndarray,
    nRowSubgrid: np.ndarray,
    nColSubgrid: np.ndarray,
) -> None:
    """Check the farthest target a PE may draw shares its subgrid.

    Targets are drawn over a rectangle anchored at the partner, so the whole
    rectangle stays in the subgrid if its far corner does.
    """
    nRow, nCol = target.shape[:2]
    rows, cols = np.indices((nRow, nCol))
    nRowSubgrid = np.broadcast_to(nRowSubgrid, (nRow, nCol)).astype(np.int64)
    nColSubgrid = np.broadcast_to(nColSubgrid, (nRow, nCol)).astype(np.int64)

    # each PE's subgrid, identified by its top-left (origin) PE
    heights = np.where(nRowSubgrid, nRowSubgrid, nRow)  # 0: whole grid
    widths = np.where(nColSubgrid, nColSubgrid, nCol)
    subgridOriginRow = rows - rows % heights
    subgridOriginCol = cols - cols % widths

    target = target.squeeze(axis=2)  # enforce depth exactly 1
    partnerSpread = np.broadcast_to(data.squeeze(axis=2), (nRow, nCol))
    has_target = target != 0xFFFFFFFF
    col = (target >> 16) + (partnerSpread >> 16)
    row = (target & 0xFFFF) + (partnerSpread & 0xFFFF)
    col, row = col[has_target], row[has_target]
    assert (col < nCol).all() and (row < nRow).all(), "spread off grid"

    is_bad = (  # is intersubgrid?
        subgridOriginRow[row, col] != subgridOriginRow[has_target]
    ) | (subgridOriginCol[row, col] != subgridOriginCol[has_target])
    assert not is_bad.any(), f"{is_bad.sum()} PEs may target another subgrid"


if __name__ == "__main__":
    comptime_config = json.loads(sys.argv[1])
    runtime_config = np.load(io.BytesIO(sys.stdin.buffer.read()))
    data = runtime_config["rtc_p2pPartnerSpread"]
    print(f"validating rtc_p2pPartnerSpread {data.shape=}")
    if "rtc_p2pTarget" not in runtime_config:
        print("no rtc_p2pTarget to offset, nothing to check")
        sys.exit()

    # subgrid sizes are runtime config too, defaulting to comptime_config
    nRowSubgrid, nColSubgrid = (
        (
            runtime_config[f"rtc_{key}"].squeeze(axis=2)
            if f"rtc_{key}" in runtime_config
            else comptime_config[key]
        )
        for key in ("nRowSubgrid", "nColSubgrid")
    )
    validate_rtc_p2pPartnerSpread_within_subgrid(
        data, runtime_config["rtc_p2pTarget"], nRowSubgrid, nColSubgrid
    )
