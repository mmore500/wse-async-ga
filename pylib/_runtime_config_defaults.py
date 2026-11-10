import numpy as np

from ._runtime_config import RuntimeConfigContext, RuntimeConfigEntry


def _make_whoami(ctx: RuntimeConfigContext) -> np.ndarray:
    return np.arange(ctx.nRow * ctx.nCol).reshape(ctx.nRow, ctx.nCol, 1)


def _make_whereami_x(ctx: RuntimeConfigContext) -> np.ndarray:
    return np.indices((ctx.nRow, ctx.nCol))[1][:, :, None]


def _make_whereami_y(ctx: RuntimeConfigContext) -> np.ndarray:
    return np.indices((ctx.nRow, ctx.nCol))[0][:, :, None]


def make_runtime_config_constant(symbol: str, value: int) -> RuntimeConfigEntry:
    """Make runtime config entry setting symbol to one value on all PEs."""
    return RuntimeConfigEntry(
        symbol, lambda ctx: np.full((ctx.nRow, ctx.nCol, 1), value)
    )


def make_runtime_config_defaults(
    ctx: RuntimeConfigContext,
) -> list[RuntimeConfigEntry]:
    """Make runtime config entries for all symbols, with default data.

    Should match defaults in kernel-async-ga/client.py.
    """
    return [
        RuntimeConfigEntry("whoami", _make_whoami),
        RuntimeConfigEntry("whereami_x", _make_whereami_x),
        RuntimeConfigEntry("whereami_y", _make_whereami_y),
        make_runtime_config_constant("globalSeed", ctx.globalSeed),
        make_runtime_config_constant("nRowSubgrid", ctx.nRowSubgrid),
        make_runtime_config_constant("nColSubgrid", ctx.nColSubgrid),
    ]
