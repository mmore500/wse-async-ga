#!/usr/bin/env python3
"""Generate a runtime config for each line of a jsonl file, from templates.

Each line is a JSON object with a `runtime_config_name` and, optionally,
comptime configs to override, e.g.
{"runtime_config_name": "seed=1", "globalSeed": 1}. Writes
runtime_config_dispatch/<runtime_config_name>.py from the template for each.
"""

import argparse
import json
import logging
import pathlib
import string

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    here = pathlib.Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "template", help="file in runtime_config_template, e.g. default.py.in"
    )
    parser.add_argument(
        "jsonl", nargs="?", default=here / "runtime_config_default.jsonl"
    )
    args = parser.parse_args()

    logging.info(
        f"generating {args.template} runtime configs from {args.jsonl}"
    )
    text = here.joinpath("runtime_config_template", args.template).read_text()
    outdir = here / "runtime_config_dispatch"
    outdir.mkdir(exist_ok=True)
    for line in pathlib.Path(args.jsonl).read_text().splitlines():
        overrides = json.loads(line)
        runtime_config_name = overrides.pop("runtime_config_name")
        outpath = outdir / f"{runtime_config_name}.py"
        if outpath.exists():
            logging.info(f"- keeping existing {outpath}")
            continue

        logging.info(f"- writing {outpath} with {overrides}")
        outpath.write_text(
            string.Template(text).substitute(
                runtime_config_name=runtime_config_name, overrides=overrides
            )
        )
