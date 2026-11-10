"""Runtime config that leaves all symbols at their defaults.

Reads kernel parameters as JSON from stdin, and writes an npz file of data to
override symbols' defaults to stdout; see README.md.
"""

import json
import sys

import numpy as np

if __name__ == "__main__":
    params = json.load(sys.stdin)
    shape = (params["nRow"], params["nCol"], 1)
    overrides = {
        # "globalSeed": np.full(shape, 42, dtype=np.uint32),
    }
    np.savez(sys.stdout.buffer, **overrides)
