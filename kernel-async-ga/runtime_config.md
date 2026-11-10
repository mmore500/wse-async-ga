# Runtime config

Runtime configs are per-PE data that `client.py` copies to device symbols
before launch. Each `<name>.py` or `<name>.npz` in `runtime_config_dispatch`
is one runtime config, and `client.py` runs once for each, in directory
`runtime_config=<name>` if there is more than one. With none there, it runs
once with default data.

- A py file is run by `client.py`, which writes `params` as JSON to its
  stdin and reads an npz file of data keyed by symbol from its stdout. It
  should check its own data, and exit nonzero if the data is bad. It can only
  import what the wheelhouse provides, and should print only to stderr.
- An npz file holds data keyed by symbol.
- Data for a symbol is an integer array of shape `(nRow, nCol, width)` with
  values that fit `uint32`. Symbols a config doesn't give keep their defaults.
- `params` has `nRow`, `nCol`, and the kernel's compile-time parameters.

`client.py` saves the data it copies to device alongside its other outputs,
as `a=runtimeconfig+name=<name>+ext=.npz`.

## Generating runtime configs

`runtime_config_generate.py` writes py files to `runtime_config_dispatch`
from templates in `runtime_config_template`, one for each line of a jsonl
file. If `runtime_config_dispatch` is missing, `compile.sh` runs it on
`runtime_config_default.jsonl`, to make `default.py`. Otherwise, run it
yourself on your own jsonl file; for example, to run seeds without
recompiling,

```bash
echo '{"name": "seed=1", "globalSeed": 1}' > sweep.jsonl
echo '{"name": "seed=2", "globalSeed": 2}' >> sweep.jsonl
./runtime_config_generate.py sweep.jsonl
```

Items other than `name` and `template` fill `${key}` placeholders in the
template; those not given take the kernel's compile-time parameter of that
name. A py file already there is kept, so you can also write one by hand.

## Validating runtime configs

Configs generated from `default.py.in` run every py file in
`runtime_config_validate/<symbol>` on their data for that symbol. Each sets
`validate(data, params)`, which returns something truthy if `data` is bad.
Add files there to add checks. A config written by hand should run its own.

A py runtime config can also be run by hand,

```bash
echo '{"nRow": 3, "nCol": 3, "globalSeed": 0, "nRowSubgrid": 0,
  "nColSubgrid": 0}' | python3 runtime_config_dispatch/default.py > default.npz
```

## Symbols

| symbol | default | effect |
| --- | --- | --- |
| `whoami` | row-major index | PE id, seeds RNG along with `globalSeed` |
| `whereami_x` | column | PE column, sets migration at grid edges |
| `whereami_y` | row | PE row, sets migration at grid edges |
| `globalSeed` | compile-time `globalSeed` | RNG seed, should match on all PEs |
| `nRowSubgrid` | compile-time `nRowSubgrid` | disables migration every n rows |
| `nColSubgrid` | compile-time `nColSubgrid` | disables migration every n cols |
