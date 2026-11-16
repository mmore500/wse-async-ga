"""Adapted from hypermutator-dynamics binder/wse-denovo-spatial2d-explicitsite-timeseries.ipynb."""

import os
import sys

import downstream
import imageio
import numpy as np
import polars as pl
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pylib.munge._addend_groups import addend_groups
from pylib.munge._squeeze import squeeze_int
from pylib.munge._union_upsample import union_upsample

NMUT = "CEREBRASLIB_HYPERMUT_NUM_AVAIL_BEN_MUTS"

data_path, outroot = sys.argv[1], sys.argv[2]
teeplot_subdir = os.path.join(outroot, "wse-denovo-spatial2d-explicitsite-timeseries")

dfxs = pl.read_parquet(data_path).with_row_index("dstream_data_id")
if NMUT not in dfxs.columns:  # not present in this dataset; placeholder
    dfxs = dfxs.with_columns(pl.lit(-1).alias(NMUT))


def unpack_logger_timeseries_data(df: pl.DataFrame) -> pl.DataFrame:
    df = df.filter((pl.col("col") % 8 == 0) & (pl.col("row") % 8 == 0))
    dfu = downstream.dataframe.unpack_data_packed(df)
    df = downstream.dataframe.explode_lookup_unpacked(
        dfu.with_columns(
            dstream_T=pl.max_horizontal(pl.col("dstream_T"), pl.col("dstream_S")),
        ),
        value_type="uint8",
    )
    df = df.join(
        dfu.select(
            "dstream_data_id",
            NMUT,
            "CEREBRASLIB_TRAITLOGGER_DILATION",
            "col",
            "replicate",
            "row",
            "tile",
        ),
        on="dstream_data_id",
    )
    df = df.with_columns(
        dstream_T=pl.col("dstream_T") * pl.col("CEREBRASLIB_TRAITLOGGER_DILATION"),
        dstream_Tbar=pl.col("dstream_Tbar")
        * pl.col("CEREBRASLIB_TRAITLOGGER_DILATION"),
    )
    df = addend_groups(
        df,
        group_by=["replicate", "tile"],
        aggs=(
            pl.col(NMUT).unique().pipe(squeeze_int),
            pl.col("col").unique().pipe(squeeze_int),
            (pl.col("dstream_Tbar") + 1).max().alias("dstream_Tbar"),
            pl.lit(0).alias("dstream_value"),
            pl.col("row").unique().pipe(squeeze_int),
        ),
        inner_only=True,
    )
    df = union_upsample(
        df,
        upsample="dstream_Tbar",
        group_by="tile",
        fill_null_ops=[
            (pl.all(), {"strategy": "forward"}),
            (pl.col("dstream_Tbar", "dstream_value"), {"strategy": "zero"}),
            (pl.all(), {"strategy": "backward"}),
        ],
    )
    assert len(df["dstream_Tbar"].value_counts()["count"].unique()) == 1
    return df


df = pl.concat(
    [unpack_logger_timeseries_data(d) for _, d in tqdm(dfxs.group_by("replicate"))]
)

os.makedirs(teeplot_subdir, exist_ok=True)
for (replicate, nmut), dfr in tqdm(df.group_by("replicate", NMUT)):
    frames = []
    for (dstream_Tbar,), dff in dfr.sort("dstream_Tbar").group_by(
        "dstream_Tbar", maintain_order=True
    ):
        dff = dff.with_columns(
            col_rank=pl.col("col").rank(method="dense").cast(pl.Int32) - 1,
            row_rank=pl.col("row").rank(method="dense").cast(pl.Int32) - 1,
        )
        num_rows = dff["row_rank"].max() + 1
        num_cols = dff["col_rank"].max() + 1
        frame = np.empty((num_rows, num_cols), dtype=np.uint8)
        frame[dff["row_rank"], dff["col_rank"]] = dff["dstream_value"] * 255
        frame = np.kron(frame, np.ones((4, 4), dtype=frame.dtype))
        frames.append(frame)

        outdir = f"{teeplot_subdir}/a=traitframes+nmut={nmut}+rep={replicate}"
        os.makedirs(outdir, exist_ok=True)
        imageio.imwrite(f"{outdir}/dstream_Tbar={dstream_Tbar:06}+ext=.png", frame.T)

    print(len(frames), "frames")
    outpath = f"{teeplot_subdir}/a=traits+nmut={nmut}+rep={replicate}+ext=.gif"
    imageio.mimsave(outpath, frames, duration=0.2)
    print(outpath)
