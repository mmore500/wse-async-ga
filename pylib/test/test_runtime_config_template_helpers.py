import importlib.util
import pathlib
import string
import types

import numpy as np
import polars as pl
import pytest

template_path = (
    pathlib.Path(__file__).resolve().parents[2]
    / "kernel-async-ga"
    / "runtime_config_template"
    / "default.py.in"
)


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> types.ModuleType:
    """Template as a module, for its helpers (its main block doesn't run)."""
    path = tmp_path_factory.mktemp("template") / "default.py"
    path.write_text(
        string.Template(template_path.read_text()).substitute(
            dispatch_slug="test", runtime_config={}
        )
    )
    spec = importlib.util.spec_from_file_location("default", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_symbol_values(template: types.ModuleType) -> None:
    shape = (3, 4, 1)
    symbols = template._eval_symbol_values(
        exprs={
            "rtc_a": "lambda shape: np.full(shape, 0.5, np.float32)",
            "rtc_b": "lambda shape: np.full(shape, -1, np.int32)",
        },
        shape=shape,
    )
    assert sorted(symbols) == ["rtc_a", "rtc_b"]
    for values in symbols.values():
        assert values.shape == shape and values.dtype == np.uint32
    assert (symbols["rtc_a"].view(np.float32) == 0.5).all()
    assert (symbols["rtc_b"] == 0xFFFFFFFF).all()  # raw bits


@pytest.mark.parametrize(
    "expr",
    [
        "lambda shape: np.zeros(shape, np.float64)",  # not 32-bit
        "lambda shape: np.zeros(shape[:2], np.float32)",  # wrong shape
    ],
)
def test_eval_symbol_values_rejects(
    template: types.ModuleType, expr: str
) -> None:
    with pytest.raises(ValueError):
        template._eval_symbol_values(exprs={"rtc_a": expr}, shape=(3, 4, 1))
    with pytest.raises(ValueError):  # not an rtc_ symbol
        template._eval_symbol_values(
            exprs={"a": "lambda shape: np.zeros(shape, np.float32)"},
            shape=(3, 4, 1),
        )


def _expected_nest_level(baseSubgridsOut: int, nestingFactor: int) -> int:
    """Largest level with nestingFactor^level at most baseSubgridsOut."""
    if nestingFactor == 1:
        return 0  # any level gives base size
    level = 0
    while nestingFactor ** (level + 1) <= baseSubgridsOut:
        level += 1
    return level


@pytest.mark.parametrize("nestingFactor", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "nRow, nCol, nRowSubgrid, nColSubgrid",
    [(32, 48, 2, 2), (40, 30, 3, 2), (17, 23, 1, 4), (9, 9, 9, 9)],
)
def test_nest_subgrid_sizes(
    template: types.ModuleType,
    nestingFactor: int,
    nRow: int,
    nCol: int,
    nRowSubgrid: int,
    nColSubgrid: int,
) -> None:
    rowSubgrid, colSubgrid = template._nest_subgrid_sizes(
        nRow=nRow,
        nCol=nCol,
        nRowSubgrid=nRowSubgrid,
        nColSubgrid=nColSubgrid,
        nestingFactor=nestingFactor,
        nestingLevelPeriod=0,
    )
    assert rowSubgrid.shape == colSubgrid.shape == (nRow, nCol)

    for row in range(nRow):
        for col in range(nCol):
            baseSubgridsOut = max(row // nRowSubgrid, col // nColSubgrid)
            level = _expected_nest_level(baseSubgridsOut, nestingFactor)
            scale = nestingFactor**level
            assert rowSubgrid[row, col] == nRowSubgrid * scale
            assert colSubgrid[row, col] == nColSubgrid * scale

    # every subgrid is aligned to its own size, and uniform in size
    rows, cols = np.indices((nRow, nCol))
    originRow = rows - rows % rowSubgrid
    originCol = cols - cols % colSubgrid
    assert (rowSubgrid[originRow, originCol] == rowSubgrid).all()
    assert (colSubgrid[originRow, originCol] == colSubgrid).all()

    if nestingFactor == 1:
        assert (rowSubgrid == nRowSubgrid).all()
        assert (colSubgrid == nColSubgrid).all()
    else:
        assert rowSubgrid.max() > nRowSubgrid or nRow <= nRowSubgrid


def test_nest_subgrid_sizes_unpartitioned(template: types.ModuleType) -> None:
    rowSubgrid, colSubgrid = template._nest_subgrid_sizes(
        nRow=5,
        nCol=6,
        nRowSubgrid=0,
        nColSubgrid=3,
        nestingFactor=1,
        nestingLevelPeriod=0,
    )
    assert (rowSubgrid == 0).all() and (colSubgrid == 3).all()


@pytest.mark.parametrize(
    "nestingFactor, nRowSubgrid, nColSubgrid",
    [(0, 2, 2), (1.5, 2, 2), (-2, 2, 2), (2, 0, 2), (2, 2, 0)],
)
def test_nest_subgrid_sizes_rejects(
    template: types.ModuleType,
    nestingFactor: int,
    nRowSubgrid: int,
    nColSubgrid: int,
) -> None:
    with pytest.raises(ValueError):
        template._nest_subgrid_sizes(
            nRow=8,
            nCol=8,
            nRowSubgrid=nRowSubgrid,
            nColSubgrid=nColSubgrid,
            nestingFactor=nestingFactor,
            nestingLevelPeriod=0,
        )


@pytest.mark.parametrize(
    "nestingFactor, nRowSubgrid, nColSubgrid, nestingLevelPeriod",
    [
        (2, 8, 8, 3),
        (2, 2, 2, 4),
        (4, 2, 2, 3),
        (3, 1, 2, 2),
        (2, 3, 5, 2),
        (1, 4, 2, 5),
        (2, 4, 4, 1),
    ],
)
@pytest.mark.parametrize("nRow, nCol", [(150, 70), (64, 64), (20, 33)])
def test_nest_subgrid_sizes_levels(
    template: types.ModuleType,
    nestingFactor: int,
    nRowSubgrid: int,
    nColSubgrid: int,
    nestingLevelPeriod: int,
    nRow: int,
    nCol: int,
) -> None:
    rowSubgrid, colSubgrid = template._nest_subgrid_sizes(
        nRow=nRow,
        nCol=nCol,
        nRowSubgrid=nRowSubgrid,
        nColSubgrid=nColSubgrid,
        nestingFactor=nestingFactor,
        nestingLevelPeriod=nestingLevelPeriod,
    )
    # each tile is the untiled pattern's corner, nestingLevelPeriod levels wide
    tileRows = nRowSubgrid * nestingFactor**nestingLevelPeriod
    tileCols = nColSubgrid * nestingFactor**nestingLevelPeriod
    tileRow, tileCol = template._nest_subgrid_sizes(
        nRow=tileRows,
        nCol=tileCols,
        nRowSubgrid=nRowSubgrid,
        nColSubgrid=nColSubgrid,
        nestingFactor=nestingFactor,
        nestingLevelPeriod=0,
    )
    rows, cols = np.indices((nRow, nCol))
    tr, tc = rows % tileRows, cols % tileCols
    assert (rowSubgrid == tileRow[tr, tc]).all()
    assert (colSubgrid == tileCol[tr, tc]).all()

    # at most nestingLevelPeriod levels
    assert rowSubgrid.max() <= nRowSubgrid * nestingFactor ** (
        nestingLevelPeriod - 1
    )

    # every subgrid is aligned to its own size, uniform in size, and within
    # one tile
    originRow = rows - rows % rowSubgrid
    originCol = cols - cols % colSubgrid
    assert (rowSubgrid[originRow, originCol] == rowSubgrid).all()
    assert (colSubgrid[originRow, originCol] == colSubgrid).all()
    assert (originRow // tileRows == rows // tileRows).all()
    assert (originCol // tileCols == cols // tileCols).all()


def test_nest_subgrid_sizes_levels_8_16_32(
    template: types.ModuleType,
) -> None:
    # 64 x 64 tiles of 4 subgrids of 8 x 8, 3 of 16 x 16, and 3 of 32 x 32
    rowSubgrid, colSubgrid = template._nest_subgrid_sizes(
        nRow=128,
        nCol=192,
        nRowSubgrid=8,
        nColSubgrid=8,
        nestingFactor=2,
        nestingLevelPeriod=3,
    )
    assert (rowSubgrid == colSubgrid).all()
    numTiles = (128 // 64) * (192 // 64)
    for size, count in ((8, 4), (16, 3), (32, 3)):
        assert (rowSubgrid == size).sum() == numTiles * count * size * size
    assert set(np.unique(rowSubgrid)) == {8, 16, 32}


@pytest.mark.parametrize(
    "nestingFactor, nRowSubgrid, nColSubgrid, nestingLevelPeriod",
    [(2, 8, 8, -1), (2, 8, 8, 1.5), (2, 0, 8, 3), (2, 8, 0, 3)],
)
def test_nest_subgrid_sizes_levels_rejects(
    template: types.ModuleType,
    nestingFactor: int,
    nRowSubgrid: int,
    nColSubgrid: int,
    nestingLevelPeriod: int,
) -> None:
    with pytest.raises(ValueError):
        template._nest_subgrid_sizes(
            nRow=128,
            nCol=128,
            nRowSubgrid=nRowSubgrid,
            nColSubgrid=nColSubgrid,
            nestingFactor=nestingFactor,
            nestingLevelPeriod=nestingLevelPeriod,
        )


def _check_pairs(
    target: np.ndarray, rowSubgrid: np.ndarray, colSubgrid: np.ndarray
) -> None:
    nRow, nCol = target.shape
    rows, cols = np.indices((nRow, nCol))
    height = np.where(rowSubgrid, rowSubgrid, nRow)
    width = np.where(colSubgrid, colSubgrid, nCol)
    originPe = (rows - rows % height) * nCol + (cols - cols % width)

    has = target != 0xFFFFFFFF
    targetRow, targetCol = target[has] & 0xFFFF, target[has] >> 16
    # target's target is self, and not self
    assert ((targetRow != rows[has]) | (targetCol != cols[has])).all()
    back = target[targetRow, targetCol]
    assert (back & 0xFFFF == rows[has]).all()
    assert (back >> 16 == cols[has]).all()
    # within subgrid
    assert (originPe[targetRow, targetCol] == originPe[has]).all()
    # all but the odd one out of each subgrid paired
    sizes = np.unique(originPe, return_counts=True)[1]
    assert has.sum() == (sizes // 2 * 2).sum()
    unpaired = np.unique(originPe[~has], return_counts=True)[1]
    assert (unpaired == 1).all()


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize(
    "nRow, nCol, nRowSubgrid, nColSubgrid, nestingFactor, nestingLevelPeriod",
    [
        (10, 12, 0, 0, 1, 0),  # one subgrid
        (3, 3, 0, 0, 1, 0),  # odd one out
        (41, 31, 3, 5, 1, 0),
        (41, 31, 3, 5, 2, 0),
        (40, 40, 2, 2, 3, 0),
        (33, 65, 1, 2, 4, 0),
        (8, 9, 1, 1, 1, 0),  # single-PE subgrids, none paired
        (130, 70, 8, 8, 2, 3),  # tiled nesting, clipped edge tiles
        (50, 50, 1, 1, 2, 4),
    ],
)
def test_pair_p2p_targets(
    template: types.ModuleType,
    seed: int,
    nRow: int,
    nCol: int,
    nRowSubgrid: int,
    nColSubgrid: int,
    nestingFactor: int,
    nestingLevelPeriod: int,
) -> None:
    rowSubgrid, colSubgrid = template._nest_subgrid_sizes(
        nRow=nRow,
        nCol=nCol,
        nRowSubgrid=nRowSubgrid,
        nColSubgrid=nColSubgrid,
        nestingFactor=nestingFactor,
        nestingLevelPeriod=nestingLevelPeriod,
    )
    target = template._pair_p2p_targets(
        rowSubgrid=rowSubgrid, colSubgrid=colSubgrid, seed=seed
    )
    assert target.shape == (nRow, nCol) and target.dtype == np.uint32
    _check_pairs(target, rowSubgrid, colSubgrid)


def test_pair_p2p_targets_seed(template: types.ModuleType) -> None:
    rowSubgrid, colSubgrid = np.full((20, 20), 5), np.full((20, 20), 4)

    def pair(seed: int) -> np.ndarray:
        return template._pair_p2p_targets(
            rowSubgrid=rowSubgrid, colSubgrid=colSubgrid, seed=seed
        )

    assert (pair(1) == pair(1)).all()
    assert (pair(1) != pair(2)).any()


def test_make_subgrid_df(template: types.ModuleType) -> None:
    rowSubgrid, colSubgrid = template._nest_subgrid_sizes(
        nRow=10,
        nCol=12,
        nRowSubgrid=2,
        nColSubgrid=3,
        nestingFactor=2,
        nestingLevelPeriod=0,
    )
    grid = template._make_subgrid_df(
        rowSubgrid=rowSubgrid, colSubgrid=colSubgrid
    )
    assert len(grid) == 10 * 12
    rows, cols = np.indices((10, 12))
    assert (grid["row"].to_numpy() == rows.ravel()).all()  # row-major
    assert (grid["col"].to_numpy() == cols.ravel()).all()
    assert (grid["nRow"] == 10).all() and (grid["nCol"] == 12).all()
    assert (grid["nRowSubgrid"].to_numpy() == rowSubgrid.ravel()).all()
    assert (grid["nColSubgrid"].to_numpy() == colSubgrid.ravel()).all()
    assert (grid["subgridOriginRow"] + grid["subgridRow"] == grid["row"]).all()
    assert (grid["subgridOriginCol"] + grid["subgridCol"] == grid["col"]).all()
    assert (grid["subgridRow"] < grid["nRowSubgrid"]).all()
    assert (grid["subgridCol"] < grid["nColSubgrid"]).all()
    assert (grid["subgridOriginRow"] % grid["nRowSubgrid"] == 0).all()


def test_make_subgrid_df_unpartitioned(template: types.ModuleType) -> None:
    grid = template._make_subgrid_df(
        rowSubgrid=np.zeros((4, 5), int), colSubgrid=np.full((4, 5), 2)
    )
    assert (grid["subgridOriginRow"] == 0).all()  # 0: whole grid
    assert (grid["subgridRow"] == grid["row"]).all()
    assert (grid["subgridOriginCol"] == grid["col"] - grid["col"] % 2).all()


def test_calc_symbol_values(template: types.ModuleType) -> None:
    shape = (3, 4, 1)
    grid = template._make_subgrid_df(
        rowSubgrid=np.full((3, 4), 3), colSubgrid=np.full((3, 4), 2)
    )
    symbols = template._calc_symbol_values(
        calcConfig={
            "with_setup": [
                "(pl.col('row') * 10 + pl.col('col')).alias('a')",
                "(pl.col('a') + 1).alias('b')",  # sees a, set before
                "pl.col('b').alias('a')",  # replaces a
            ],
            "with_values": {
                "rtc_a": "pl.col('a').cast(pl.UInt32)",
                "rtc_f": "(pl.col('subgridCol') / 2).cast(pl.Float32)",
                "rtc_i": "(-pl.col('row')).cast(pl.Int32)",
            },
        },
        subgrid_df=grid,
        shape=shape,
    )
    assert symbols.keys() == {"rtc_a", "rtc_f", "rtc_i"}
    for values in symbols.values():
        assert values.shape == shape and values.dtype == np.uint32
    rows, cols = np.indices((3, 4))
    assert (symbols["rtc_a"][..., 0] == rows * 10 + cols + 1).all()
    assert (symbols["rtc_f"][..., 0].view(np.float32) == (cols % 2) / 2).all()
    assert (symbols["rtc_i"][..., 0].view(np.int32) == -rows).all()

    assert (
        template._calc_symbol_values(
            calcConfig={}, subgrid_df=grid, shape=shape
        )
        == {}
    )


@pytest.mark.parametrize(
    "calcConfig",
    [
        {"with_values": {"a": "pl.col('row').cast(pl.UInt32)"}},  # not rtc_
        {"with_values": {"rtc_a": "pl.col('row')"}},  # 64-bit
        {"with_values": {"rtc_a": "pl.col('row').head(2).cast(pl.UInt32)"}},
        {"with_setup": [], "with_symbols": {}},  # unknown key
        {"with_setup": ["pl.col('nope').alias('a')"]},  # no such column
    ],
)
def test_calc_symbol_values_rejects(
    template: types.ModuleType, calcConfig: dict
) -> None:
    grid = template._make_subgrid_df(
        rowSubgrid=np.full((3, 4), 3), colSubgrid=np.full((3, 4), 2)
    )
    with pytest.raises((ValueError, pl.exceptions.PolarsError)):
        template._calc_symbol_values(
            calcConfig=calcConfig, subgrid_df=grid, shape=(3, 4, 1)
        )
