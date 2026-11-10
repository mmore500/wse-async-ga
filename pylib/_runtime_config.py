import dataclasses
import typing
import warnings

import numpy as np


@dataclasses.dataclass(frozen=True)
class RuntimeConfigContext:
    """Compiled kernel parameters available to runtime config entries."""

    nRow: int
    nCol: int
    nRowSubgrid: int = 0
    nColSubgrid: int = 0
    globalSeed: int = 0


@dataclasses.dataclass(frozen=True)
class RuntimeConfigRule:
    """A validation rule for one runtime config entry.

    Calling `check` with an entry's data and the context should return
    something truthy or nonempty (e.g., the offending items) on failure.
    """

    name: str
    check: typing.Callable[[np.ndarray, RuntimeConfigContext], typing.Any]
    mode: typing.Literal["strict", "warn", "log"] = "strict"


@dataclasses.dataclass(frozen=True)
class RuntimeConfigEntry:
    """Per-PE data to copy from host to a device symbol before launch.

    Calling `generate` with the context should return data for all PEs, as an
    integer array with shape (nRow, nCol, width) and values that fit uint32.
    """

    symbol: str
    generate: typing.Callable[[RuntimeConfigContext], np.ndarray]
    rules: typing.Sequence[RuntimeConfigRule] = ()


def build_runtime_config(
    ctx: RuntimeConfigContext, entries: typing.Iterable[RuntimeConfigEntry]
) -> dict[str, np.ndarray]:
    """Generate and validate data for runtime config entries.

    Raises, warns, or prints for each rule that fails, per its mode.

    Returns
    -------
    dict[str, np.ndarray]
        Arrays with dtype uint32 and shape (nRow, nCol, width), keyed by
        symbol.
    """
    result = {}
    for entry in entries:
        symbol = entry.symbol
        if symbol in result:
            raise ValueError(f"more than one entry for {symbol}")

        data = np.asarray(entry.generate(ctx))
        if data.ndim != 3 or data.shape[:2] != (ctx.nRow, ctx.nCol):
            raise ValueError(f"{symbol} has bad shape {data.shape}")
        if not np.issubdtype(data.dtype, np.integer):
            raise ValueError(f"{symbol} has non-integer dtype {data.dtype}")
        if data.size and not 0 <= data.min() <= data.max() <= 0xFFFFFFFF:
            raise ValueError(f"{symbol} has values that don't fit in uint32")

        for rule in entry.rules:
            failure = rule.check(data, ctx)
            msg = f"{symbol} fails rule {rule.name}:\n{failure}"
            if not (len(failure) if hasattr(failure, "__len__") else failure):
                pass
            elif rule.mode == "strict":
                raise ValueError(msg)
            elif rule.mode == "warn":
                warnings.warn(msg)
            elif rule.mode == "log":
                print(msg)
            else:
                raise ValueError(f"unknown {rule.mode=}")

        result[symbol] = data.astype(np.uint32)

    return result
