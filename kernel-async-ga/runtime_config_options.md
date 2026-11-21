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
