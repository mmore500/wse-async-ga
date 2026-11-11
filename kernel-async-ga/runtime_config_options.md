# Runtime config options

Items that a jsonl line may set, besides `dispatch_slug`, to generate a
runtime config from `runtime_config_template/default.py.in` with
`runtime_config_generate.py`. Options that set a device symbol are only
passed if set; otherwise, the symbol keeps its default.

| option               | device symbol            | default                  | effect                                |
| -------------------- | ------------------------ | ------------------------ | ------------------------------------- |
| `globalSeed`         | `rtc_globalSeed`         | comptime `globalSeed`    | RNG seed, same on all PEs             |
| `nRowSubgrid`        | `rtc_nRowSubgrid`        | comptime `nRowSubgrid`   | no migration every n rows, 0 for none |
| `nColSubgrid`        | `rtc_nColSubgrid`        | comptime `nColSubgrid`   | no migration every n cols, 0 for none |
| `nCycleAtLeast`      | `rtc_nCycleAtLeast`      | comptime `nCycleAtLeast` | how many cycles to run for            |
| `p2pSendProbability` | `rtc_p2pSendProbability` | comptime probability     | chance per cycle to send to target    |
| `p2pMode`            | `rtc_p2pTarget`          | `"none"`                 | point-to-point targets, see below     |
| `p2pSeed`            |                          | `globalSeed`             | seeds random choice of targets        |

Point-to-point exchange has to be compiled in for `p2pSendProbability` and
`p2pMode` to have effect, by setting a nonzero compile-time probability
`COMPCONFENV_ASYNC_GA_P2P_SEND_PROBABILITY__f32`.

| `p2pMode` | targets                                                              |
| --------- | -------------------------------------------------------------------- |
| `"none"`  | no PE has a target                                                   |
| `"pairs"` | PEs target each other in random pairs within subgrid, one may lack   |
