# runtime_config

Runtime configs are per-PE data that `client.py` copies to device symbols
before launch.

- `<name>.py` or `<name>.npz` is one runtime config. `client.py` runs once
  for each here, in directory `runtime_config=<name>` if there is more than
  one. Symbols a config doesn't give keep their defaults.
  - A py file is run by `client.py`, which writes `params` as JSON to its
    stdin and reads an npz file of data keyed by symbol from its stdout.
    It should check its own data, and exit nonzero if the data is bad. It can
    only import what the wheelhouse provides, and should print only to
    stderr.
  - An npz file holds data keyed by symbol.
  - Data for a symbol is an integer array of shape `(nRow, nCol, width)`
    with values that fit `uint32`.
- `params` has `nRow`, `nCol`, and the kernel's compile-time parameters.

A py runtime config can also be run by hand,

```bash
echo '{"nRow": 3, "nCol": 3}' | python3 default.py > default.npz
```

`client.py` saves the data it copies to device alongside its other outputs,
as `a=runtimeconfig+name=<name>+ext=.npz`.

`default.py` overrides nothing. To make others, write one by hand or see
`pyscript/runtime-config-make-*.py`.

| symbol | default | effect |
| --- | --- | --- |
| `whoami` | row-major index | PE id, seeds RNG along with `globalSeed` |
| `whereami_x` | column | PE column, sets migration at grid edges |
| `whereami_y` | row | PE row, sets migration at grid edges |
| `globalSeed` | compile-time `globalSeed` | RNG seed, should match on all PEs |
| `nRowSubgrid` | compile-time `nRowSubgrid` | disables migration every n rows |
| `nColSubgrid` | compile-time `nColSubgrid` | disables migration every n cols |
