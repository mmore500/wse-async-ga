#!/usr/bin/env python3
"""Write one runtime config for each value of a symbol, `<symbol>=<value>`.

Each sets the symbol to that value on all PEs, and leaves others at
defaults; e.g., `--symbol globalSeed --values 1 2 3` to run replicates
without recompiling.

Runtime configs are py files generating per-PE data for device symbols; see
kernel-async-ga/runtime_config/README.md.
"""

import argparse
import os

template = '''\
"""Runtime config setting {symbol} to {value} on all PEs."""

import numpy as np


def generate(params: dict) -> "dict[str, np.ndarray]":
    shape = (params["nRow"], params["nCol"], 1)
    return {{"{symbol}": np.full(shape, {value}, dtype=np.uint32)}}
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--values", type=int, nargs="+", required=True)
    parser.add_argument("-o", "--outdir", default="runtime_config")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    for value in args.values:
        path = os.path.join(args.outdir, f"{args.symbol}={value}.py")
        with open(path, "w", encoding="utf-8") as py_file:
            py_file.write(template.format(symbol=args.symbol, value=value))
        print(f"wrote {path}")
