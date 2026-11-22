# Schema for runtime config jsonl entries

Each line of a jsonl file passed to `runtime_config_generate.py` generates
one runtime config from `runtime_config_template/default.py.in`.
If no runtime config value is specified, symbols keep compile time default.

- `dispatch_slug` (required)
  - names the runtime config, its script
    `runtime_config_dispatch/<dispatch_slug>.py`, and its output folder
    `dispatch=<dispatch_slug>`
- `globalSeed`
  - RNG seed, same on all PEs
  - sets `rtc_globalSeed`
- `nRowSubgrid`
  - no migration every n rows, 0 for none
  - sets `rtc_nRowSubgrid`
- `nColSubgrid`
  - no migration every n cols, 0 for none
  - sets `rtc_nColSubgrid`
- `nestingFactorSubgrid`
  - positive integer, default 1 (uniform subgrids); if more, subgrids grow by
    this factor outward from PE (0, 0), e.g., 2 with b x b base subgrids
    (`nRowSubgrid` x `nColSubgrid`):
    ```
    +---+---+-------+---------------+
    | b | b |       |               |
    +---+---+  2b   |               |
    | b | b |       |               |
    +---+---+-------+      4b       |
    |       |       |               |
    |  2b   |  2b   |               |
    |       |       |               |
    +-------+-------+---------------+
    |               |               |
    |               |               |
    |               |               |
    |      4b       |      4b       |
    |               |               |
    |               |               |
    |               |               |
    +---------------+---------------+
    ```
  - sets `rtc_nRowSubgrid` and `rtc_nColSubgrid` per PE
- `nCycleAtLeast`
  - how many cycles to run for
  - sets `rtc_nCycleAtLeast`
- `p2pSendProbability`
  - chance per cycle to send to p2p target
  - sets `rtc_p2pSendProbability`
- `p2pMode`
  - how p2p targets are assigned
  - `"none"` (default): no PE has a target, `rtc_p2pTarget` is not set
  - `"pairs"`: PEs target each other in random pairs within subgrid, one
    per subgrid may be left unpaired; sets `rtc_p2pTarget`
- `p2pSeed`
  - seeds random choice of targets
  - if not specified, `globalSeed`
- `setSymbolValues`
  - device symbols set from a Python lambda of symbol `shape` (nRow, nCol,
    1) giving 32-bit values, e.g., `f32`, `i32`, or `u32`:
    `{"rtc_foo": "lambda shape: np.full(shape, 0.5, np.float32)"}`
  - each symbol must be exported (e.g., via genome `export_symbol_names`)
- `calcSymbolValues`
  - device symbols calculated from Python strings of polars expressions
    (`pl`, and `np`, in scope) over a per-PE grid frame, with columns `row`,
    `col`, `nRow`, `nCol`, `nRowSubgrid`, `nColSubgrid` (each PE's subgrid
    size, after any nesting; 0 for none), `subgridOriginRow` and
    `subgridOriginCol` (subgrid's top-left PE), and `subgridRow` and
    `subgridCol` (PE position within subgrid)
  - `with_setup`: list of expressions, each adding or replacing frame
    columns via `with_columns`, in order (so each sees those before)
  - `with_values`: dict of symbol to expression giving its 32-bit per-PE
    values, e.g., cast to `pl.UInt32`, `pl.Int32`, or `pl.Float32`:
    ```
    {"with_setup": ["(pl.col('subgridRow') == 0).alias('isTop')"],
     "with_values": {"rtc_foo": "pl.col('isTop').cast(pl.UInt32)"}}
    ```
  - evaluated last, after any other option, so sees final subgrid sizes;
    each symbol must be exported (e.g., via genome `export_symbol_names`)
    and not set by any other option
  - recipe, a parameter set per subgrid size:
    ```
    {"with_values": {"rtc_foo": "pl.col('nRowSubgrid').replace_strict({9: 0.5, 18: 0.25}, return_dtype=pl.Float32)"}}
    ```
  - recipe, parameter sets rotated evenly over each subgrid size's
    replicate subgrids (number each size's subgrids, then take modulo):
    ```
    {"with_setup": [
       "(pl.col('subgridOriginRow') * pl.col('nCol') + pl.col('subgridOriginCol')).alias('subgrid')",
       "(pl.col('subgrid').rank('dense').over('nRowSubgrid', 'nColSubgrid') - 1).alias('replicate')",
       "(pl.col('replicate') % 3).alias('paramSet')"],
     "with_values": {"rtc_foo": "pl.col('paramSet').replace_strict({0: 0.1, 1: 0.2, 2: 0.3}, return_dtype=pl.Float32)"}}
    ```