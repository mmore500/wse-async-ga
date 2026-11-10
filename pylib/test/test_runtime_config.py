import pathlib

import numpy as np
import pytest

from pylib._runtime_config import (
    RuntimeConfigContext,
    RuntimeConfigEntry,
    RuntimeConfigRule,
    build_runtime_config,
    save_runtime_configs,
)


def _make_whoami(ctx: RuntimeConfigContext) -> np.ndarray:
    return np.arange(ctx.nRow * ctx.nCol).reshape(ctx.nRow, ctx.nCol, 1)


def _find_duplicates(data: np.ndarray, ctx: RuntimeConfigContext) -> list:
    values, counts = np.unique(data, return_counts=True)
    return values[counts > 1].tolist()


def test_build_runtime_config() -> None:
    ctx = RuntimeConfigContext(nRow=3, nCol=4)
    rule = RuntimeConfigRule("unique", _find_duplicates)
    res = build_runtime_config(
        ctx,
        [
            RuntimeConfigEntry("whoami", _make_whoami, rules=[rule]),
            RuntimeConfigEntry("two", lambda ctx: np.ones((3, 4, 2), np.uint8)),
        ],
    )
    assert [*res] == ["whoami", "two"]
    assert res["whoami"].dtype == res["two"].dtype == np.uint32
    assert res["whoami"].shape == (3, 4, 1)
    assert res["two"].shape == (3, 4, 2)
    assert res["whoami"].ravel().tolist() == [*range(12)]


@pytest.mark.parametrize(
    "generate",
    [
        lambda ctx: np.zeros((ctx.nRow, ctx.nCol)),  # missing width axis
        lambda ctx: np.zeros((ctx.nCol, ctx.nRow, 1), np.uint32),  # transposed
        lambda ctx: np.zeros((ctx.nRow, ctx.nCol, 1), np.float32),
        lambda ctx: np.full((ctx.nRow, ctx.nCol, 1), -1),
    ],
)
def test_build_runtime_config_bad_data(generate: callable) -> None:
    ctx = RuntimeConfigContext(nRow=3, nCol=4)
    with pytest.raises(ValueError):
        build_runtime_config(ctx, [RuntimeConfigEntry("whoami", generate)])


def test_build_runtime_config_duplicate_symbol() -> None:
    ctx = RuntimeConfigContext(nRow=3, nCol=4)
    entry = RuntimeConfigEntry("whoami", _make_whoami)
    with pytest.raises(ValueError):
        build_runtime_config(ctx, [entry, entry])


def test_build_runtime_config_rule_modes(
    capsys: pytest.CaptureFixture,
) -> None:
    ctx = RuntimeConfigContext(nRow=3, nCol=4)

    def build(mode: str) -> dict:
        rule = RuntimeConfigRule("unique", _find_duplicates, mode)
        entry = RuntimeConfigEntry(
            "whoami", lambda ctx: np.zeros((3, 4, 1), np.uint32), rules=[rule]
        )
        return build_runtime_config(ctx, [entry])

    with pytest.raises(ValueError, match="whoami fails rule unique"):
        build("strict")
    with pytest.warns(UserWarning, match="whoami fails rule unique"):
        assert [*build("warn")] == ["whoami"]
    assert [*build("log")] == ["whoami"]
    assert "whoami fails rule unique" in capsys.readouterr().out
    with pytest.raises(ValueError, match="unknown"):
        build("bogus")


def test_save_runtime_configs(tmp_path: pathlib.Path) -> None:
    ctx = RuntimeConfigContext(nRow=3, nCol=4)
    configs = {
        f"globalSeed={seed}": build_runtime_config(
            ctx,
            [
                RuntimeConfigEntry(
                    "globalSeed",
                    lambda ctx, seed=seed: np.full((3, 4, 1), seed),
                )
            ],
        )
        for seed in (1, 2)
    }
    save_runtime_configs(configs, str(tmp_path / "runtime_config"))
    for seed in (1, 2):
        path = tmp_path / "runtime_config" / f"globalSeed={seed}.npz"
        with np.load(path) as npz_file:
            assert npz_file.files == ["globalSeed"]
            assert (npz_file["globalSeed"] == seed).all()
            assert npz_file["globalSeed"].dtype == np.uint32
