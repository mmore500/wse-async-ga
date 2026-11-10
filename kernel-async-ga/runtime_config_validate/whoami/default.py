"""Checks data for whoami, in runtime config npz file from stdin."""

import io
import sys

import numpy as np

if __name__ == "__main__":
    data = np.load(io.BytesIO(sys.stdin.buffer.read()))["whoami"]
    print(f"validating whoami {data.shape=}")
    assert len(np.unique(data)) == data.size, "PE ids are not unique"
