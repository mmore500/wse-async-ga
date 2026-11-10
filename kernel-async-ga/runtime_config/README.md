# runtime_config

Runtime configs are per-PE data that `client.py` copies to device symbols
before launch.

- `<name>.npz` holds one runtime config, with an array of shape
  `(nRow, nCol, width)` and dtype `uint32` per symbol. `client.py` runs once
  for each npz file here, in directory `runtime_config=<name>` if there is
  more than one. Symbols missing from a file keep their defaults.
- `validate_<symbol>.py`, if present, is run by `client.py` each time data
  for that symbol is loaded from an npz file. It should set
  `validate(data, params)`, which returns something truthy if `data` is bad;
  `params` has `nRow`, `nCol`, and the kernel's compile-time parameters.

`compile.sh` writes `default.npz`. To make others, see
`pyscript/runtime-config-*.py`.

| symbol | default | effect |
| --- | --- | --- |
| `whoami` | row-major index | PE id, seeds RNG along with `globalSeed` |
| `whereami_x` | column | PE column, sets migration at grid edges |
| `whereami_y` | row | PE row, sets migration at grid edges |
| `globalSeed` | compile-time `globalSeed` | RNG seed, should match on all PEs |
| `nRowSubgrid` | compile-time `nRowSubgrid` | disables migration every n rows |
| `nColSubgrid` | compile-time `nColSubgrid` | disables migration every n cols |
