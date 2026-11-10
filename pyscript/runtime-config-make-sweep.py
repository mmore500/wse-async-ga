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
"""Runtime config setting {symbol} to {value} on all PEs.

Reads kernel parameters as JSON from stdin, and writes an npz file of data to
override symbols' defaults to stdout; see README.md.
"""

import json
import sys

import numpy as np

if __name__ == "__main__":
    params = json.load(sys.stdin)
    shape = (params["nRow"], params["nCol"], 1)
    {symbol} = np.full(shape, {value}, dtype=np.uint32)
{validation}
    np.savez(sys.stdout.buffer, {symbol}={symbol})
'''

# checks for config to run on its data, by symbol
validations = {
    "globalSeed": """
    # kernel computes local seeds as globalSeed * nRow * nCol + whoami + 1
    if (int(globalSeed.max()) + 1) * globalSeed.size >= 2**32:
        sys.exit("seed too large, local seeds would overflow u32")
""",
}

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
            py_file.write(
                template.format(
                    symbol=args.symbol,
                    value=value,
                    validation=validations.get(args.symbol, ""),
                )
            )
        print(f"wrote {path}")
