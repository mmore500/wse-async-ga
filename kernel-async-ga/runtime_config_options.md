# Schema for runtime config jsonl entries

Each line of a jsonl file passed to `runtime_config_generate.py` generates
one runtime config from `runtime_config_template/default.py.in`. Items that
set a device symbol are only passed if present; otherwise, the symbol keeps
its default.

- `dispatch_slug` (required): names the runtime config, its script
  `runtime_config_dispatch/<dispatch_slug>.py`, and its output folder
  `dispatch=<dispatch_slug>`
- `globalSeed`: RNG seed, same on all PEs
  - sets `rtc_globalSeed`, default comptime `globalSeed`
- `nRowSubgrid`: no migration every n rows, 0 for none
  - sets `rtc_nRowSubgrid`, default comptime `nRowSubgrid`
- `nColSubgrid`: no migration every n cols, 0 for none
  - sets `rtc_nColSubgrid`, default comptime `nColSubgrid`
- `nCycleAtLeast`: how many cycles to run for
  - sets `rtc_nCycleAtLeast`, default comptime `nCycleAtLeast`
- `p2pSendProbability`: chance per cycle to send to point-to-point target
  - sets `rtc_p2pSendProbability`, default comptime probability
- `p2pMode`: how point-to-point targets are assigned, default `"none"`
  - `"none"`: no PE has a target, `rtc_p2pTarget` is not set
  - `"pairs"`: PEs target each other in random pairs within subgrid, one
    may be left without; sets `rtc_p2pTarget`
- `p2pSeed`: seeds random choice of targets, default `globalSeed`

Point-to-point exchange has to be compiled in for `p2pSendProbability` and
`p2pMode` to have effect, by setting a nonzero compile-time probability
`COMPCONFENV_ASYNC_GA_P2P_SEND_PROBABILITY__f32`.
