import io
import json
import pathlib
import string
import subprocess
import sys

import numpy as np
import pytest

kernel_dir = pathlib.Path(__file__).resolve().parents[2] / "kernel-async-ga"
validate_dir = kernel_dir / "runtime_config_validate" / "rtc_p2pPartner"


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
    data = np.load(io.BytesIO(npz))["rtc_p2pPartner"]
    assert data.shape == (nRow, nCol, 2) and data.dtype == np.uint32
    assert (data[:, :, 0] != 0xFFFFFFFF).sum() == numPaired

    # pairs: partner's partner is self
    rows, cols = np.indices((nRow, nCol))
    is_paired = data[:, :, 0] != 0xFFFFFFFF
    back = data[data[:, :, 1][is_paired], data[:, :, 0][is_paired]]
    assert (back[:, 0] == cols[is_paired]).all()
    assert (back[:, 1] == rows[is_paired]).all()

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
    # (0, 0) <-> (1, 1) paired, others unpaired; values are (col, row)
    good = np.array(
        [[[1, 1], [none, none]], [[none, none], [0, 0]]], dtype=np.uint32
    )
    validators = "default.py", "intersubgrid.py"
    assert all(
        _validate(v, _to_npz(rtc_p2pPartner=good), comptime) for v in validators
    )

    # pairs are one mode of connectivity, others pass validators too
    one_way = good.copy()
    one_way[1, 1] = none
    assert all(
        _validate(v, _to_npz(rtc_p2pPartner=one_way), comptime)
        for v in validators
    )

    off_grid = good.copy()
    off_grid[0, 1] = (5, 0)
    assert not _validate(
        "default.py", _to_npz(rtc_p2pPartner=off_grid), comptime
    )

    # subgrid size comes from runtime config if there, else comptime
    assert not _validate(
        "intersubgrid.py",
        _to_npz(rtc_p2pPartner=good),
        {**comptime, "nRowSubgrid": 1},
    )
    assert not _validate(
        "intersubgrid.py",
        _to_npz(
            rtc_p2pPartner=good,
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
