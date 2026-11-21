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
- `nestedSubgridSizeFactor`
  - integer, default 1 (uniform `nRowSubgrid` x `nColSubgrid` subgrids)
  - if more than 1, subgrids nest outward from PE (0, 0): smallest subgrids
    (`nRowSubgrid` x `nColSubgrid`, which must be equal and nonzero) at the
    corner, then L-shaped shells of `nestedSubgridSizeFactor`^2 - 1 subgrids at each
    size `nRowSubgrid` * `nestedSubgridSizeFactor`^k, until the grid is covered;
    subgrids overhanging grid edges are clipped
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
- `uniformConstant`
  - device symbols set to the same value on every PE, as `type` `f32`,
    `i32`, or `u32`, e.g., `{"rtc_foo": {"value": 0.5, "type": "f32"}}`
  - each symbol must be listed in `cerebraslib/rtc_named_exports.csl`
