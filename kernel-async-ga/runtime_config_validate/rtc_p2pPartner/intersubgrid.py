"""Validator that rtc_p2pPartner values in a runtime config stay in subgrids.

Takes comptime configs json as argument and runtime config npz on stdin.
"""

import io
import json
import sys

import numpy as np


def validate_rtc_p2pPartner_within_subgrid(
    data: np.ndarray, nRowSubgrid: int, nColSubgrid: int
) -> None:
    nRow, nCol = data.shape[:2]
    rows, cols = np.indices((nRow, nCol))
    is_paired = data[:, :, 0] != 0xFFFFFFFF
    is_intersubgrid = is_paired & (
        (
            data[:, :, 1] // (nRowSubgrid or nRow)
            != rows // (nRowSubgrid or nRow)
        )
        | (
            data[:, :, 0] // (nColSubgrid or nCol)
            != cols // (nColSubgrid or nCol)
        )
    )
    num_bad = is_intersubgrid.sum()
    assert not num_bad, f"{num_bad} PEs have partner in another subgrid"


if __name__ == "__main__":
    comptime_config = json.loads(sys.argv[1])
    runtime_config = np.load(io.BytesIO(sys.stdin.buffer.read()))
    data = runtime_config["rtc_p2pPartner"]
    print(f"validating rtc_p2pPartner {data.shape=}")

    # subgrid sizes are runtime config too, defaulting to comptime_config
    nRowSubgrid, nColSubgrid = (
        (
            int(runtime_config[f"rtc_{key}"].max())
            if f"rtc_{key}" in runtime_config
            else comptime_config[key]
        )
        for key in ("nRowSubgrid", "nColSubgrid")
    )
    print(f"{nRowSubgrid=} {nColSubgrid=}")
    validate_rtc_p2pPartner_within_subgrid(data, nRowSubgrid, nColSubgrid)
