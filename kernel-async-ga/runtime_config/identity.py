"""Runtime config that sets PE identity symbols to their default values.

Example for pyscript/runtime-config-build.py `--spec`.
"""

import numpy as np

from pylib._runtime_config import (
    RuntimeConfigContext,
    RuntimeConfigEntry,
    RuntimeConfigRule,
)


def make_whoami(ctx: RuntimeConfigContext) -> np.ndarray:
    return np.arange(ctx.nRow * ctx.nCol).reshape(ctx.nRow, ctx.nCol, 1)


def make_whereami_x(ctx: RuntimeConfigContext) -> np.ndarray:
    return np.indices((ctx.nRow, ctx.nCol))[1][:, :, None]


def make_whereami_y(ctx: RuntimeConfigContext) -> np.ndarray:
    return np.indices((ctx.nRow, ctx.nCol))[0][:, :, None]


def find_duplicates(data: np.ndarray, ctx: RuntimeConfigContext) -> np.ndarray:
    values, counts = np.unique(data, return_counts=True)
    return values[counts > 1]


RUNTIME_CONFIG = [
    RuntimeConfigEntry(
        "whoami",
        make_whoami,
        rules=[RuntimeConfigRule("unique", find_duplicates)],
    ),
    RuntimeConfigEntry("whereami_x", make_whereami_x),
    RuntimeConfigEntry("whereami_y", make_whereami_y),
]
