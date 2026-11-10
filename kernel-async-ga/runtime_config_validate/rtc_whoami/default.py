"""Validators for rtc_whoami values in a runtime config.

Takes comptime configs json as argument and runtime config npz on stdin.
"""

import io
import json
import sys

import numpy as np


def validate_rtc_whoami_unique(data: np.ndarray, comptime: dict) -> None:
    assert len(np.unique(data)) == data.size, "PE ids are not unique"


if __name__ == "__main__":
    comptime = json.loads(sys.argv[1])
    data = np.load(io.BytesIO(sys.stdin.buffer.read()))["rtc_whoami"]
    print(f"validating rtc_whoami {data.shape=}")
    validate_rtc_whoami_unique(data, comptime)
