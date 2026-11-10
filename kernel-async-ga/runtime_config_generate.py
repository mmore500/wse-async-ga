#!/usr/bin/env python3
"""Generate a runtime config for each line of a jsonl file, from templates.

Writes runtime_config_dispatch/<name>.py for each line, which is a JSON
object with a `name` and, optionally, a `template` in runtime_config_template
and comptime configs to override, e.g. {"name": "seed=1", "globalSeed": 1}.
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
        "jsonl", nargs="?", default=here / "runtime_config_default.jsonl"
    )
    args = parser.parse_args()

    logging.info(f"generating runtime configs from {args.jsonl}")
    outdir = here / "runtime_config_dispatch"
    outdir.mkdir(exist_ok=True)
    for line in pathlib.Path(args.jsonl).read_text().splitlines():
        overrides = json.loads(line)
        name = overrides.pop("name")
        template = overrides.pop("template", "default.py.in")
        outpath = outdir / f"{name}.py"
        if outpath.exists():
            logging.info(f"- keeping existing {outpath}")
            continue

        logging.info(f"- writing {outpath} from {template} with {overrides}")
        text = here.joinpath("runtime_config_template", template).read_text()
        outpath.write_text(
            string.Template(text).substitute(name=name, overrides=overrides)
        )
