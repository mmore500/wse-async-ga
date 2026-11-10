#!/usr/bin/env python3
"""Write one runtime config for each value of a symbol, `<symbol>=<value>`.

Sets the symbol to that value on all PEs, and leaves others at defaults; e.g.,
`--symbol globalSeed --values 1 2 3` to run replicates without recompiling.

Runtime configs are npz files of per-PE data for device symbols; see
kernel-async-ga/runtime_config/README.md.
"""

import argparse

from pylib._runtime_config import (
    build_runtime_config,
    load_runtime_config_context,
    save_runtime_configs,
)
from pylib._runtime_config_defaults import make_runtime_config_constant

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compile_json", help="out.json written by cslc")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--values", type=int, nargs="+", required=True)
    parser.add_argument("-o", "--outdir", default="runtime_config")
    args = parser.parse_args()

    ctx = load_runtime_config_context(args.compile_json)
    save_runtime_configs(
        {
            f"{args.symbol}={value}": build_runtime_config(
                ctx, [make_runtime_config_constant(args.symbol, value)]
            )
            for value in args.values
        },
        args.outdir,
    )
