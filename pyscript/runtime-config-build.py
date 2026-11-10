#!/usr/bin/env python3
"""Build runtime config for a compiled kernel-async-ga.

Generates and validates per-PE data for device symbols, and writes it as an
npz file keyed by symbol. Put the file alongside kernel-async-ga/client.py as
`runtime_config.npz` to have it copied to device before launch.

Symbols without data in the file keep the defaults set by client.py. Pass
`--spec` a Python file that sets `RUNTIME_CONFIG` to a list of
`RuntimeConfigEntry`; see kernel-async-ga/runtime_config for examples.
"""

import argparse
import json
import runpy

import numpy as np

from pylib._runtime_config import RuntimeConfigContext, build_runtime_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "compile_json", help="out.json written by cslc, e.g. out/out.json"
    )
    parser.add_argument(
        "--spec", required=True, help="Python file setting RUNTIME_CONFIG"
    )
    parser.add_argument("-o", "--output", required=True, help="npz file")
    args = parser.parse_args()

    with open(args.compile_json, encoding="utf-8") as json_file:
        params = json.load(json_file)["params"]
    ctx = RuntimeConfigContext(
        nRow=int(params["nRow"]),
        nCol=int(params["nCol"]),
        nRowSubgrid=int(params["nRowSubgrid"]),
        nColSubgrid=int(params["nColSubgrid"]),
        globalSeed=int(params["globalSeed"]),
    )
    print(ctx)

    entries = runpy.run_path(args.spec)["RUNTIME_CONFIG"]
    runtime_config = build_runtime_config(ctx, entries)
    for symbol, data in runtime_config.items():
        print(f"{symbol} {data.shape}")

    with open(args.output, "wb") as npz_file:
        np.savez(npz_file, **runtime_config)
    print(f"wrote {len(runtime_config)} symbols to {args.output}")
