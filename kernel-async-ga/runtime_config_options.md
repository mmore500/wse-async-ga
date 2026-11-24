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
- `nestingLevelPeriodSubgrid`
  - if L > 0, repeats the first L nesting levels as tiles over the PE array
  - default 0, no repeat
  - e.g., `nestingFactorSubgrid` 2 with L 2, b = `nRowSubgrid`:
    ```
    +---+---+-------+---+---+-------+
    | b | b |       | b | b |       |
    +---+---+  2b   +---+---+  2b   |
    | b | b |       | b | b |       |
    +---+---+-------+---+---+-------+
    |       |       |       |       |
    |  2b   |  2b   |  2b   |  2b   |
    |       |       |       |       |
    +-------+-------+-------+-------+
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
- `p2pPartnerSpread`
  - how far past its partner a PE may draw its send target, packed as
    `col << 16 | row`; each axis is drawn uniformly over `[0, spread]`, anew
    for every send
  - `0` (default): sends to the partner exactly
  - validators check the farthest target stays on the grid and in the
    sending PE's subgrid
  - sets `rtc_p2pPartnerSpread`
- `setSymbolValues`
  - device symbols set from a Python lambda of symbol `shape` (nRow, nCol,
    1) giving 32-bit values, e.g., `f32`, `i32`, or `u32`:
    `{"rtc_foo": "lambda shape: np.full(shape, 0.5, np.float32)"}`
  - each symbol must be exported (e.g., via genome `export_symbol_names`)
- `calcSymbolValues`
  - device symbols calculated from polars expression strings over a per-PE
    dataframe (see `_make_grid_df` in `runtime_config_template/default.py.in`)
  - `with_setup`: list of expressions to set up helper columns, evaluated sequentially
  - `with_values`: dict mapping symbols to polars expressions
    `{"rtc_foo": "pl.col('nRowSubgrid').cast(pl.Float32) / 8"}`
