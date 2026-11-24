import io
import json
import pathlib
import string
import subprocess
import sys
from collections import Counter

import numpy as np
import pytest

kernel_dir = pathlib.Path(__file__).resolve().parents[2] / "kernel-async-ga"
validate_dir = kernel_dir / "runtime_config_validate" / "rtc_p2pTarget"


def _generate(
    tmp_path: pathlib.Path, comptime: dict, runtime_config: dict
) -> bytes:
    """Run runtime config from default.py.in template, giving its npz."""
    template = kernel_dir / "runtime_config_template" / "default.py.in"
    script = tmp_path / "default.py"
    script.write_text(
        string.Template(template.read_text()).substitute(
            dispatch_slug="test", runtime_config=runtime_config
        )
    )
    return subprocess.run(
        [sys.executable, script],
        input=json.dumps(comptime).encode(),
        stdout=subprocess.PIPE,
        check=True,
    ).stdout


def _validate(validator: str, npz: bytes, comptime: dict) -> bool:
    """Does npz pass validator?"""
    return not subprocess.run(
        [sys.executable, validate_dir / validator, json.dumps(comptime)],
        input=npz,
        capture_output=True,
        check=False,
    ).returncode


def _to_npz(**runtime_config: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **runtime_config)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "nRow, nCol, nRowSubgrid, nColSubgrid, numPaired",
    [(3, 3, 0, 0, 8), (5, 7, 2, 3, 32), (6, 6, 3, 0, 36), (1, 1, 0, 0, 0)],
)
@pytest.mark.parametrize("seed", [0, 1])
def test_p2p_pairs_template(
    tmp_path: pathlib.Path,
    nRow: int,
    nCol: int,
    nRowSubgrid: int,
    nColSubgrid: int,
    numPaired: int,
    seed: int,
) -> None:
    comptime = {
        "nRow": nRow,
        "nCol": nCol,
        "nRowSubgrid": nRowSubgrid,
        "nColSubgrid": nColSubgrid,
        "globalSeed": seed,
        "nCycleAtLeast": 10,
        "p2pSendProbability": 0.5,
    }
    pairs = {"p2pMode": "pairs"}
    npz = _generate(tmp_path, comptime, pairs)
    data = np.load(io.BytesIO(npz))["rtc_p2pTarget"]
    assert data.shape == (nRow, nCol, 1) and data.dtype == np.uint32
    data = data[:, :, 0]  # col << 16 | row
    has_target = data != 0xFFFFFFFF
    assert has_target.sum() == numPaired

    # pairs: target's target is self
    rows, cols = np.indices((nRow, nCol))
    back = data[data[has_target] & 0xFFFF, data[has_target] >> 16]
    assert (back >> 16 == cols[has_target]).all()
    assert (back & 0xFFFF == rows[has_target]).all()

    assert npz == _generate(tmp_path, comptime, pairs)
    for validator in "default.py", "intersubgrid.py":
        assert _validate(validator, npz, comptime), validator


def test_p2p_pairs_template_seed(tmp_path: pathlib.Path) -> None:
    comptime = {
        "nRow": 6,
        "nCol": 6,
        "nRowSubgrid": 0,
        "nColSubgrid": 0,
        "globalSeed": 0,
        "nCycleAtLeast": 10,
        "p2pSendProbability": 0.5,
    }
    pairs = {"p2pMode": "pairs"}
    npz = _generate(tmp_path, comptime, pairs)
    assert npz != _generate(tmp_path, {**comptime, "globalSeed": 1}, pairs)
    assert npz != _generate(tmp_path, comptime, {**pairs, "globalSeed": 1})
    assert npz != _generate(tmp_path, comptime, {**pairs, "p2pSeed": 1})


def test_p2p_validators_reject() -> None:
    comptime = {"nRow": 2, "nCol": 2, "nRowSubgrid": 0, "nColSubgrid": 0}
    none = 0xFFFFFFFF
    # (0, 0) <-> (1, 1) paired, others lack target; values are col << 16 | row
    good = np.array([[[1 << 16 | 1], [none]], [[none], [0]]], dtype=np.uint32)
    validators = "default.py", "intersubgrid.py"
    assert all(
        _validate(v, _to_npz(rtc_p2pTarget=good), comptime) for v in validators
    )

    # pairs are one mode of connectivity, others pass validators too
    one_way = good.copy()
    one_way[1, 1] = none
    assert all(
        _validate(v, _to_npz(rtc_p2pTarget=one_way), comptime)
        for v in validators
    )

    off_grid = good.copy()
    off_grid[0, 1] = 5 << 16
    assert not _validate(
        "default.py", _to_npz(rtc_p2pTarget=off_grid), comptime
    )

    # subgrid size comes from runtime config if there, else comptime
    assert not _validate(
        "intersubgrid.py",
        _to_npz(rtc_p2pTarget=good),
        {**comptime, "nRowSubgrid": 1},
    )
    assert not _validate(
        "intersubgrid.py",
        _to_npz(
            rtc_p2pTarget=good,
            rtc_nRowSubgrid=np.zeros((2, 2, 1), np.uint32),
            rtc_nColSubgrid=np.ones((2, 2, 1), np.uint32),
        ),
        comptime,
    )


def test_default_template_p2p_modes(tmp_path: pathlib.Path) -> None:
    comptime = {
        "nRow": 4,
        "nCol": 4,
        "nRowSubgrid": 0,
        "nColSubgrid": 0,
        "globalSeed": 0,
        "nCycleAtLeast": 10,
        "p2pSendProbability": 0.25,
    }
    # symbols that runtime config doesn't set are left at comptime config
    assert not np.load(io.BytesIO(_generate(tmp_path, comptime, {}))).files

    runtime_config = np.load(
        io.BytesIO(_generate(tmp_path, comptime, {"p2pSendProbability": 0.5}))
    )
    assert runtime_config.files == ["rtc_p2pSendProbability"]
    probability = runtime_config["rtc_p2pSendProbability"].view(np.float32)
    assert (probability == 0.5).all()

    with pytest.raises(subprocess.CalledProcessError):
        _generate(tmp_path, comptime, {"p2pMode": "bogus"})


@pytest.mark.parametrize(
    "probability, ok", [(0.0, True), (1.0, True), (2.0, False)]
)
def test_p2p_send_probability_validator(probability: float, ok: bool) -> None:
    npz = _to_npz(
        rtc_p2pSendProbability=np.full((2, 2, 1), probability, np.float32).view(
            np.uint32
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            validate_dir.with_name("rtc_p2pSendProbability") / "default.py",
            json.dumps({"p2pSendProbability": 1.0}),
        ],
        input=npz,
        capture_output=True,
        check=False,
    )
    assert (result.returncode == 0) == ok


@pytest.mark.parametrize(
    "runtime, comptime, ok",
    [(0.0, 0.0, True), (0.5, 0.5, True), (0.0, 0.5, True), (0.5, 0.0, False)],
)
def test_p2p_send_probability_validator_compiled_in(
    runtime: float, comptime: float, ok: bool
) -> None:
    npz = _to_npz(
        rtc_p2pSendProbability=np.full((2, 2, 1), runtime, np.float32).view(
            np.uint32
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            validate_dir.with_name("rtc_p2pSendProbability") / "default.py",
            json.dumps({"p2pSendProbability": comptime}),
        ],
        input=npz,
        capture_output=True,
        check=False,
    )
    assert (result.returncode == 0) == ok


def _validate_subgrid(axis: str, npz: bytes, comptime: dict) -> bool:
    """Does npz pass rtc_n{axis}Subgrid validator?"""
    return not subprocess.run(
        [
            sys.executable,
            validate_dir.with_name(f"rtc_n{axis}Subgrid") / "default.py",
            json.dumps(comptime),
        ],
        input=npz,
        capture_output=True,
        check=False,
    ).returncode


@pytest.mark.parametrize("growth", [2, 3, 4])
@pytest.mark.parametrize(
    "nRow, nCol, baseRow, baseCol",
    [(30, 19, 1, 1), (70, 45, 2, 2), (70, 45, 3, 2)],
)
def test_subgrid_growth_template(
    tmp_path: pathlib.Path,
    growth: int,
    nRow: int,
    nCol: int,
    baseRow: int,
    baseCol: int,
) -> None:
    comptime = {
        "nRow": nRow,
        "nCol": nCol,
        "nRowSubgrid": baseRow,
        "nColSubgrid": baseCol,
        "globalSeed": 0,
    }
    runtime_config = {"nestingFactorSubgrid": growth, "p2pMode": "pairs"}
    npz = _generate(tmp_path, comptime, runtime_config)
    data = np.load(io.BytesIO(npz))
    height = data["rtc_nRowSubgrid"][:, :, 0].astype(int)
    width = data["rtc_nColSubgrid"][:, :, 0].astype(int)
    assert (height * baseCol == width * baseRow).all()  # same nesting level

    # subgrids aligned to their own size, so PEs agree on walls
    rows, cols = np.indices((nRow, nCol))
    top, left = rows - rows % height, cols - cols % width
    for row, col in zip(*np.indices((nRow, nCol)).reshape(2, -1)):
        h, w = height[row, col], width[row, col]
        t, l = top[row, col], left[row, col]
        assert (height[t : t + h, l : l + w] == h).all()
        assert (width[t : t + h, l : l + w] == w).all()

    # smallest at corner, then growth^2 - 1 subgrids per size, growing out
    assert baseRow * growth**2 <= nRow and baseCol * growth**2 <= nCol
    corners = {(t, l, h) for t, l, h in zip(top.flat, left.flat, height.flat)}
    full = Counter(
        h
        for t, l, h in corners
        if t + h <= nRow and l + h * baseCol // baseRow <= nCol
    )
    assert height[0, 0] == baseRow and full[baseRow] == growth**2
    assert full[baseRow * growth] == growth**2 - 1

    for axis in "Row", "Col":
        assert _validate_subgrid(axis, npz, comptime)
    for validator in "default.py", "intersubgrid.py":
        assert _validate(validator, npz, comptime), validator

    # p2p pairs stay within (variable-size) subgrids
    target = data["rtc_p2pTarget"][:, :, 0]
    has = target != 0xFFFFFFFF
    trow, tcol = target[has] & 0xFFFF, target[has] >> 16
    assert (top[trow, tcol] == top[has]).all()
    assert (left[trow, tcol] == left[has]).all()
    assert has.sum() >= nRow * nCol - len(corners)


def test_subgrid_growth_one_is_uniform(tmp_path: pathlib.Path) -> None:
    comptime = {
        "nRow": 9,
        "nCol": 7,
        "nRowSubgrid": 2,
        "nColSubgrid": 3,
        "globalSeed": 0,
    }
    pairs = {"p2pMode": "pairs"}
    assert _generate(tmp_path, comptime, pairs) == _generate(
        tmp_path, comptime, {**pairs, "nestingFactorSubgrid": 1}
    )
    for bad in (
        {"nestingFactorSubgrid": 0},
        {"nestingFactorSubgrid": 1.5},
    ):
        with pytest.raises(subprocess.CalledProcessError):
            _generate(tmp_path, comptime, bad)
    with pytest.raises(subprocess.CalledProcessError):  # no base subgrids
        _generate(
            tmp_path,
            {**comptime, "nColSubgrid": 0},
            {"nestingFactorSubgrid": 2},
        )


def test_calc_symbol_values_template(tmp_path: pathlib.Path) -> None:
    comptime = {
        "nRow": 12,
        "nCol": 10,
        "nRowSubgrid": 2,
        "nColSubgrid": 2,
        "globalSeed": 0,
    }
    runtime_config = {
        "nestingFactorSubgrid": 2,  # sees nested sizes
        "calcSymbolValues": {
            "with_setup": [
                (
                    "(pl.col('nRowSubgrid') * pl.col('nColSubgrid'))"
                    ".alias('area')"
                ),
            ],
            "with_values": {
                "rtc_area": "pl.col('area').cast(pl.UInt32)",
                "rtc_isOrigin": "((pl.col('subgridRow') == 0)"
                " & (pl.col('subgridCol') == 0)).cast(pl.UInt32)",
            },
        },
    }
    data = np.load(io.BytesIO(_generate(tmp_path, comptime, runtime_config)))
    height = data["rtc_nRowSubgrid"][:, :, 0].astype(int)
    width = data["rtc_nColSubgrid"][:, :, 0].astype(int)
    area = data["rtc_area"]
    assert area.shape == (12, 10, 1) and area.dtype == np.uint32
    assert (area[:, :, 0] == height * width).all()
    rows, cols = np.indices((12, 10))
    isOrigin = (rows % height == 0) & (cols % width == 0)
    assert (data["rtc_isOrigin"][:, :, 0] == isOrigin).all()

    for bad in (  # double sets symbol from another option
        {
            "nestingFactorSubgrid": 2,
            "calcSymbolValues": {
                "with_values": {
                    "rtc_nRowSubgrid": "pl.col('row').cast(pl.UInt32)",
                },
            },
        },
        {
            "setSymbolValues": {
                "rtc_a": "lambda shape: np.zeros(shape, np.uint32)",
            },
            "calcSymbolValues": {
                "with_values": {"rtc_a": "pl.col('row').cast(pl.UInt32)"},
            },
        },
    ):
        with pytest.raises(subprocess.CalledProcessError):
            _generate(tmp_path, comptime, bad)


# recipes: subgrid replicate index within each subgrid size, so parameter
# sets can rotate evenly over each size's replicate subgrids
_REPLICATE_SETUP = [
    (
        "(pl.col('subgridOriginRow') * pl.col('nCol')"
        " + pl.col('subgridOriginCol')).alias('subgrid')"
    ),
    (
        "(pl.col('subgrid').rank('dense').over('nRowSubgrid', 'nColSubgrid')"
        " - 1).alias('replicate')"
    ),
]


def _nested_comptime(nRow: int, nCol: int) -> dict:
    return {
        "nRow": nRow,
        "nCol": nCol,
        "nRowSubgrid": 2,
        "nColSubgrid": 2,
        "globalSeed": 0,
    }


def _subgrid_values(data: np.lib.npyio.NpzFile, symbol: str) -> dict:
    """Map each subgrid (top, left, size) to its PEs' one value."""
    height = data["rtc_nRowSubgrid"][:, :, 0].astype(int)
    width = data["rtc_nColSubgrid"][:, :, 0].astype(int)
    values = data[symbol][:, :, 0]
    nRow, nCol = height.shape
    rows, cols = np.indices((nRow, nCol))
    top, left = rows - rows % height, cols - cols % width
    res = {}
    for t, l, h, v in zip(top.flat, left.flat, height.flat, values.flat):
        assert res.setdefault((t, l, h), v) == v  # uniform within subgrid
    return res


@pytest.mark.parametrize("nRow, nCol", [(32, 32), (40, 24), (64, 48)])
def test_calc_symbol_values_recipe_even_sets(
    tmp_path: pathlib.Path, nRow: int, nCol: int
) -> None:
    # parameter sets (here, two symbols each) rotated evenly over each
    # subgrid size's replicate subgrids
    sets = [(0.1, 7), (0.2, 8), (0.3, 9)]
    runtime_config = {
        "nestingFactorSubgrid": 2,
        "calcSymbolValues": {
            "with_setup": [
                *_REPLICATE_SETUP,
                f"(pl.col('replicate') % {len(sets)}).alias('paramSet')",
            ],
            "with_values": {
                "rtc_rate": "pl.col('paramSet').replace_strict("
                f"{dict(enumerate(r for r, _ in sets))},"
                " return_dtype=pl.Float32)",
                "rtc_count": "pl.col('paramSet').replace_strict("
                f"{dict(enumerate(c for _, c in sets))},"
                " return_dtype=pl.UInt32)",
            },
        },
    }
    data = np.load(
        io.BytesIO(
            _generate(tmp_path, _nested_comptime(nRow, nCol), runtime_config)
        )
    )
    rates = _subgrid_values(data, "rtc_rate")
    counts = _subgrid_values(data, "rtc_count")
    for size in {h for _, _, h in rates}:
        sub = [k for k in rates if k[2] == size]
        tally = Counter(
            (float(rates[k].view(np.float32)), int(counts[k])) for k in sub
        )
        # only listed sets, as evenly as possible within each size
        assert tally.keys() <= {(np.float32(r), c) for r, c in sets}
        assert len(tally) == min(len(sets), len(sub))
        assert max(tally.values()) - min(tally.values()) <= 1


@pytest.mark.parametrize("nRow, nCol", [(32, 32), (40, 24)])
def test_calc_symbol_values_recipe_sets_by_size(
    tmp_path: pathlib.Path, nRow: int, nCol: int
) -> None:
    # a parameter set per subgrid size
    rateBySize = {2: 0.5, 4: 0.25, 8: 0.125, 16: 0.0625, 32: 0.03125}
    runtime_config = {
        "nestingFactorSubgrid": 2,
        "calcSymbolValues": {
            "with_values": {
                "rtc_rate": "pl.col('nRowSubgrid').replace_strict("
                f"{rateBySize}, return_dtype=pl.Float32)",
            },
        },
    }
    data = np.load(
        io.BytesIO(
            _generate(tmp_path, _nested_comptime(nRow, nCol), runtime_config)
        )
    )
    height = data["rtc_nRowSubgrid"][:, :, 0].astype(int)
    rate = data["rtc_rate"][:, :, 0].view(np.float32)
    assert set(np.unique(height)) <= rateBySize.keys()
    for size, r in rateBySize.items():
        assert (rate[height == size] == np.float32(r)).all()


@pytest.mark.parametrize(
    "nRow, nCol, base, growth, levels",
    [(130, 70, 8, 2, 3), (40, 24, 2, 2, 2), (36, 36, 1, 3, 2)],
)
def test_subgrid_levels_template(
    tmp_path: pathlib.Path,
    nRow: int,
    nCol: int,
    base: int,
    growth: int,
    levels: int,
) -> None:
    comptime = {
        "nRow": nRow,
        "nCol": nCol,
        "nRowSubgrid": base,
        "nColSubgrid": base,
        "globalSeed": 0,
    }
    runtime_config = {
        "nestingFactorSubgrid": growth,
        "nestingLevelPeriodSubgrid": levels,
        "p2pMode": "pairs",
    }
    npz = _generate(tmp_path, comptime, runtime_config)
    data = np.load(io.BytesIO(npz))
    height = data["rtc_nRowSubgrid"][:, :, 0].astype(int)
    width = data["rtc_nColSubgrid"][:, :, 0].astype(int)
    assert (height == width).all()

    # each tile repeats the corner tile, with at most levels levels
    tile = base * growth**levels
    rows, cols = np.indices((nRow, nCol))
    assert (height == height[rows % tile, cols % tile]).all()
    assert height[0, 0] == base
    assert height.max() == base * growth ** (levels - 1)

    for axis in "Row", "Col":
        assert _validate_subgrid(axis, npz, comptime)
    for validator in "default.py", "intersubgrid.py":
        assert _validate(validator, npz, comptime), validator

    # p2p pairs stay within subgrids, so within tiles
    top, left = rows - rows % height, cols - cols % width
    target = data["rtc_p2pTarget"][:, :, 0]
    has = target != 0xFFFFFFFF
    trow, tcol = target[has] & 0xFFFF, target[has] >> 16
    assert (top[trow, tcol] == top[has]).all()
    assert (left[trow, tcol] == left[has]).all()
    assert (trow // tile == rows[has] // tile).all()


def test_subgrid_levels_rejects(tmp_path: pathlib.Path) -> None:
    comptime = {
        "nRow": 64,
        "nCol": 64,
        "nRowSubgrid": 8,
        "nColSubgrid": 8,
        "globalSeed": 0,
    }
    for bad in (
        {"nestingFactorSubgrid": 2, "nestingLevelPeriodSubgrid": -1},
        {"nestingFactorSubgrid": 2, "nestingLevelPeriodSubgrid": 1.5},
    ):
        with pytest.raises(subprocess.CalledProcessError):
            _generate(tmp_path, comptime, bad)


def test_subgrid_validators_reject() -> None:
    comptime = {"nRow": 4, "nCol": 4, "nRowSubgrid": 0, "nColSubgrid": 0}
    uniform = np.full((4, 4, 1), 2, np.uint32)
    for axis in "Row", "Col":
        assert _validate_subgrid(
            axis, _to_npz(**{f"rtc_n{axis}Subgrid": uniform}), comptime
        )
    # size 2 PEs above/left of size 3 PEs disagree on walls
    mixed = uniform.copy()
    mixed[2:, 2:] = 3
    for axis in "Row", "Col":
        assert not _validate_subgrid(
            axis, _to_npz(**{f"rtc_n{axis}Subgrid": mixed}), comptime
        )


def test_named_runtime_configs(tmp_path: pathlib.Path) -> None:
    comptime = {
        "nRow": 3,
        "nCol": 2,
        "nRowSubgrid": 0,
        "nColSubgrid": 0,
        "globalSeed": 0,
    }
    runtime_config = {
        "setSymbolValues": {
            "rtc_foo": "lambda shape: np.full(shape, 2.0**-20, np.float32)",
            "rtc_bar": "lambda shape: np.full(shape, -3, np.int32)",
            "rtc_baz": "lambda shape: np.full(shape, 7, np.uint32)",
            "rtc_qux": "lambda shape: np.arange(6, dtype=np.uint32)"
            ".reshape(shape)",
        },
    }
    data = np.load(io.BytesIO(_generate(tmp_path, comptime, runtime_config)))
    assert sorted(data.files) == ["rtc_bar", "rtc_baz", "rtc_foo", "rtc_qux"]
    for symbol, dtype, value in (
        ("rtc_foo", np.float32, 2.0**-20),
        ("rtc_bar", np.int32, -3),
        ("rtc_baz", np.uint32, 7),
        ("rtc_qux", np.uint32, np.arange(6).reshape(3, 2, 1)),
    ):
        assert data[symbol].shape == (3, 2, 1)
        assert data[symbol].dtype == np.uint32  # raw bits, as client copies
        assert (data[symbol].view(dtype) == value).all()

    for bad in (
        {"setSymbolValues": {"rtc_foo": "lambda s: np.full(s, 1.0)"}},
        {"setSymbolValues": {"rtc_foo": "lambda s: np.zeros(2, np.float32)"}},
        {"setSymbolValues": {"foo": "lambda s: np.zeros(s, np.float32)"}},
        {  # set twice
            "nRowSubgrid": 2,
            "setSymbolValues": {
                "rtc_nRowSubgrid": "lambda s: np.full(s, 2, np.uint32)",
            },
        },
    ):
        with pytest.raises(subprocess.CalledProcessError):
            _generate(tmp_path, comptime, bad)


@pytest.mark.parametrize(
    "rate, ok",
    [(2.0**-16, True), (0.0, True), (-1.0, False), (np.nan, False)],
)
def test_rtc_evoepi_mutation_rate_validator(rate: float, ok: bool) -> None:
    npz = _to_npz(
        rtc_evoepi_mutationRate=np.full((2, 2, 1), rate, np.float32).view(
            np.uint32
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            validate_dir.with_name("rtc_evoepi_mutationRate") / "default.py",
            "{}",
        ],
        input=npz,
        capture_output=True,
        check=False,
    )
    assert (result.returncode == 0) == ok


def _validate_spread(validator: str, npz: bytes, comptime: dict) -> bool:
    """Does npz pass an rtc_p2pPartnerSpread validator?"""
    return not subprocess.run(
        [
            sys.executable,
            validate_dir.with_name("rtc_p2pPartnerSpread") / validator,
            json.dumps(comptime),
        ],
        input=npz,
        capture_output=True,
        check=False,
    ).returncode


@pytest.mark.parametrize("partnerSpread", [0, 1, 2 << 16 | 3])
def test_p2p_partner_spread_template(
    tmp_path: pathlib.Path, partnerSpread: int
) -> None:
    comptime = {
        "nRow": 4,
        "nCol": 4,
        "nRowSubgrid": 0,
        "nColSubgrid": 0,
        "globalSeed": 0,
        "nCycleAtLeast": 10,
        "p2pSendProbability": 0.5,
    }
    # unset by default, so the kernel's own default (no offset) stands
    npz = _generate(tmp_path, comptime, {"p2pMode": "pairs"})
    assert "rtc_p2pPartnerSpread" not in np.load(io.BytesIO(npz)).files

    npz = _generate(
        tmp_path,
        comptime,
        {"p2pMode": "pairs", "p2pPartnerSpread": partnerSpread},
    )
    data = np.load(io.BytesIO(npz))["rtc_p2pPartnerSpread"]
    assert data.shape == (4, 4, 1) and data.dtype == np.uint32
    assert (data == partnerSpread).all()


def test_p2p_partner_spread_validators_reject() -> None:
    comptime = {"nRow": 4, "nCol": 4, "nRowSubgrid": 0, "nColSubgrid": 0}
    none = 0xFFFFFFFF
    # (0, 0) <-> (1, 1) paired, others lack target; values are col << 16 | row
    target = np.full((4, 4, 1), none, np.uint32)
    target[0, 0], target[1, 1] = 1 << 16 | 1, 0

    def npz(partnerSpread: int, **kwargs: np.ndarray) -> bytes:
        return _to_npz(
            rtc_p2pTarget=target,
            rtc_p2pPartnerSpread=np.full((4, 4, 1), partnerSpread, np.uint32),
            **kwargs,
        )

    for validator in "default.py", "intersubgrid.py":
        assert _validate_spread(validator, npz(0), comptime), validator
        assert _validate_spread(validator, npz(2 << 16 | 2), comptime)
        # (1, 1) plus 3 falls off the 4 x 4 grid
        assert not _validate_spread(validator, npz(3 << 16), comptime)
        assert not _validate_spread(validator, npz(3), comptime)

    # spread alone is bounded by the grid, with or without targets
    assert not _validate_spread(
        "default.py",
        _to_npz(rtc_p2pPartnerSpread=np.full((4, 4, 1), 4, np.uint32)),
        comptime,
    )
    assert _validate_spread(
        "intersubgrid.py",
        _to_npz(rtc_p2pPartnerSpread=np.full((4, 4, 1), 4, np.uint32)),
        comptime,
    )  # nothing to check without targets

    # within the grid, but reaching out of a 2 x 2 subgrid
    assert _validate_spread("default.py", npz(1 << 16 | 1), comptime)
    assert not _validate_spread(
        "intersubgrid.py",
        npz(1 << 16 | 1),
        {**comptime, "nRowSubgrid": 2, "nColSubgrid": 2},
    )
    assert not _validate_spread(
        "intersubgrid.py",
        npz(
            1 << 16 | 1,
            rtc_nRowSubgrid=np.full((4, 4, 1), 2, np.uint32),
            rtc_nColSubgrid=np.full((4, 4, 1), 2, np.uint32),
        ),
        comptime,
    )
