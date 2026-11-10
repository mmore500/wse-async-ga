#!/usr/bin/env python3
"""Write runtime config `default`, with default data for all symbols.

Runtime configs are npz files of per-PE data for device symbols; see
kernel-async-ga/runtime_config/README.md.
"""

import argparse

from pylib._runtime_config import (
    build_runtime_config,
    load_runtime_config_context,
    save_runtime_configs,
)
from pylib._runtime_config_defaults import make_runtime_config_defaults

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compile_json", help="out.json written by cslc")
    parser.add_argument("-o", "--outdir", default="runtime_config")
    args = parser.parse_args()

    ctx = load_runtime_config_context(args.compile_json)
    entries = make_runtime_config_defaults(ctx)
    save_runtime_configs(
        {"default": build_runtime_config(ctx, entries)}, args.outdir
    )
