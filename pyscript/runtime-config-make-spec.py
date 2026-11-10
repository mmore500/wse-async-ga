#!/usr/bin/env python3
"""Write runtime configs set out by a Python spec file.

The spec file should set `RUNTIME_CONFIGS` to a dict of lists of
`RuntimeConfigEntry`, keyed by runtime config name. It is run with
`ctx` set to the `RuntimeConfigContext` for the compiled kernel.

Unlike py runtime configs, the spec is run here rather than by client.py, so
can use pylib, and data is written as npz files. See
kernel-async-ga/runtime_config/README.md.
"""

import argparse
import runpy

from pylib._runtime_config import (
    build_runtime_config,
    load_runtime_config_context,
    save_runtime_configs,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compile_json", help="out.json written by cslc")
    parser.add_argument("spec", help="Python file setting RUNTIME_CONFIGS")
    parser.add_argument("-o", "--outdir", default="runtime_config")
    args = parser.parse_args()

    ctx = load_runtime_config_context(args.compile_json)
    specs = runpy.run_path(args.spec, {"ctx": ctx})["RUNTIME_CONFIGS"]
    save_runtime_configs(
        {
            name: build_runtime_config(ctx, entries)
            for name, entries in specs.items()
        },
        args.outdir,
    )
