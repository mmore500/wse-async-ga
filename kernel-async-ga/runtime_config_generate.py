#!/usr/bin/env python3
"""Generate runtime configs listed in jsonl files, from templates.

Each line of a jsonl file is one runtime config, as a JSON object with its
`name` and, optionally, its `template` in runtime_config_template (default
`default.py.in`). Other items are substituted into the template for `${key}`
placeholders; those without an item are substituted to take the kernel's
compile-time parameter of that name.

Writes runtime_config_dispatch/<name>.py, unless it is already there. See
runtime_config.md.
"""

import argparse
import json
import pathlib
import re
import string

here = pathlib.Path(__file__).resolve().parent


class Substitutions(dict):
    def __missing__(self, key: str) -> str:
        return f'params["{key}"]'  # compile-time param, read by config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "jsonl",
        nargs="*",
        default=[here / "runtime_config_default.jsonl"],
        help="files listing runtime configs to generate",
    )
    parser.add_argument(
        "-o", "--outdir", default=here / "runtime_config_dispatch"
    )
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    for jsonl in args.jsonl:
        lines = pathlib.Path(jsonl).read_text(encoding="utf-8").splitlines()
        for config in map(json.loads, filter(str.strip, lines)):
            name = config["name"]
            if not re.fullmatch(r"[\w=+.-]+", name):
                raise ValueError(f"runtime config {name=} is not a slug")

            outpath = outdir / f"{name}.py"
            if outpath.exists():
                print(f"kept {outpath}")
                continue

            template = here.joinpath(
                "runtime_config_template",
                config.get("template", "default.py.in"),
            ).read_text(encoding="utf-8")
            outpath.write_text(
                string.Template(template).substitute(Substitutions(config)),
                encoding="utf-8",
            )
            print(f"wrote {outpath}")
