# Runtime config

Per-PE data that `client.py` copies to device symbols before launch.

- `runtime_config_dispatch/<name>.py` is one runtime config. `client.py` runs
  once for each, in directory `runtime_config=<name>` if there are several.
  It gets kernel parameters as JSON on stdin, and writes an npz file of
  `uint32` arrays with shape `(nRow, nCol, 1)`, keyed by symbol, to stdout.
- `runtime_config_validate/<symbol>/*.py` are run by `client.py` on each
  runtime config. They get its npz file on stdin, and raise if data is bad.
- `runtime_config_generate.py [jsonl]` writes a runtime config for each line
  of a jsonl file, filling `runtime_config_template/default.py.in`. A line's
  items override kernel parameters. `compile.sh` runs it on
  `runtime_config_default.jsonl` if `runtime_config_dispatch` is missing.

```bash
echo '{"name": "seed=1", "globalSeed": 1}' > sweep.jsonl
echo '{"name": "seed=2", "globalSeed": 2}' >> sweep.jsonl
./runtime_config_generate.py sweep.jsonl
```

## Symbols

| symbol        | default                    | effect                           |
| ------------- | -------------------------- | -------------------------------- |
| `whoami`      | `row * nCol + col`         | PE id, seeds RNG with globalSeed |
| `whereami_x`  | `col`                      | PE column, to find grid edges    |
| `whereami_y`  | `row`                      | PE row, to find grid edges       |
| `globalSeed`  | compile-time `globalSeed`  | RNG seed, same on all PEs        |
| `nRowSubgrid` | compile-time `nRowSubgrid` | no migration every n rows        |
| `nColSubgrid` | compile-time `nColSubgrid` | no migration every n cols        |

A symbol missing from a runtime config, or all symbols if there are no
runtime configs, gets its default.
