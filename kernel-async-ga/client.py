print("kernel-async-ga/client.py ############################################")
print("######################################################################")
import argparse
import atexit
from collections import Counter
import functools
import gc
import itertools as it
import json
import logging
import os
import pathlib
import random
import uuid
import shutil
import subprocess
import sys
import time

logging.basicConfig(
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
)


def log(msg: str, *args, **kwargs) -> None:
    msg = str(msg).replace("\n", "\n" + " " * 29)
    logging.info(msg, *args, **kwargs)


@functools.lru_cache(maxsize=None)
def log_once(msg: str, *args, **kwargs) -> bool:
    before = log_once.cache_info().hits
    log(msg, *args, **kwargs)
    return log_once.cache_info().hits != before


def removeprefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


# adapted from https://docs.python.org/3/library/itertools.html#itertools.batched
def batched(iterable, n: int):
    # batched('ABCDEFG', 3) → ABC DEF G
    iterator = iter(iterable)
    while True:
        batch = tuple(tqdm(it.islice(iterator, n), desc="loading batch"))
        if not batch:
            break
        yield batch


def assemble_binary_data(
    raw_binary_data: "np.ndarray",
    nWav: int,
    verbose: bool = False,
) -> "np.ndarray":
    if verbose:
        log(f"- begin assemble_binary_data...")
        log(f"  - raw_binary_data.dtype={raw_binary_data.dtype}")
        log(f"  - raw_binary_data.shape={raw_binary_data.shape}")
        log(f"  - raw_binary_data.flat[:nWav]={raw_binary_data.flat[:nWav]}")
        log(f"  - nWav={nWav} verbose={verbose}")

        for word in range(nWav):
            log(f"---------------------------------------- binary word {word}")
            values = (
                inner[word] for outer in raw_binary_data for inner in outer
            )
            log(str([*it.islice(values, 10)]))

    binary_ints = np.ascontiguousarray(raw_binary_data.astype(">u4").ravel())
    assert binary_ints.shape == (nRow * nCol * nWav,)
    if verbose:
        log("------------------------------------------------ binary u32 ints")
        for binary_int in binary_ints[:10]:
            log(f"{len(binary_ints)=} {binary_int=}")

    binary_strings = binary_ints.view(f"V{nWav * 4}")
    assert binary_strings.shape == (nRow * nCol,)
    if verbose:
        log("------------------------------------------------- binary strings")
        log(f"  - target dtype: V{nWav * 4}")
        for binary_string in binary_strings[:10]:
            log(f"{binary_string=}")

    return binary_strings


def assemble_genome_data(
    data: "np.ndarray", verbose: bool = False
) -> "np.ndarray":
    return assemble_binary_data(data, nWav=nWav, verbose=verbose)


def assemble_genome_bookend_data(
    data: "np.ndarray", verbose: bool = False
) -> "np.ndarray":
    return assemble_binary_data(data, nWav=nWav + 2, verbose=verbose)


def process_fossils(nWav: int) -> None:
    log("reading fossils ----------------------------------------------------")
    file_size_gb = os.path.getsize("raw/fossils.npz") / (1024 * 1024 * 1024)
    log(f"- raw/fossils.npz file size: {file_size_gb:.2f} GB")

    with np.load("raw/fossils.npz") as fossils:
        layer_T = sorted(map(int, fossils.files))
        log("- done!")

        if layer_T:
            log("example fossil ---------------------------------------------")
            example_fossil = fossils[str(layer_T[0])]

            fossil_filename = "a=rawfossildat+i=0+ext=.npy"
            log(f"- saving {fossil_filename}...")
            np.save(fossil_filename, example_fossil)

            log(f"- ... saved {fossil_filename}!")

            file_size_mb = os.path.getsize(fossil_filename) / (1024 * 1024)
            log(f"- {fossil_filename} file size: {file_size_mb:.2f} MB")

            log("- example assembly")
            assemble_genome_bookend_data(example_fossil, verbose=True)

            del example_fossil
            gc.collect()

        log("assembling fossils ---------------------------------------------")
        assembled_fossils = []
        for i, fossil_batch in tqdm(
            enumerate(batched((fossils[str(i)] for i in layer_T), 100)),
            desc="fossil batches",
            total=(len(layer_T) + 99) // 100,
        ):
            log(f" - batch {i=} {len(fossil_batch)=}")

            log(f"- map assemble_genome_bookend_data over {len(fossil_batch)=}...")
            work = map(assemble_genome_bookend_data, fossil_batch)
            assembled_fossils.extend(
                tqdm(work, total=len(fossil_batch), desc="assembling fossils"),
            )
            del fossil_batch
            gc.collect()

    fossils = assembled_fossils
    del assembled_fossils
    gc.collect()

    log("dataframing fossils ------------------------------------------------")
    log(f" - {len(fossils)=}")
    if fossils:
        log(" - concatenating fossils")
        fossils = np.concatenate(fossils)
        gc.collect()

        byte_width = (nWav + 2) * 4
        log(f" - creating contiguous fossil bytes {byte_width=}")
        fossils = np.ascontiguousarray(
            fossils.view(np.uint8).reshape(-1, byte_width),
        )
        gc.collect()

        log(" - creating pyarrow view of fossils")
        fossils = pa.FixedSizeBinaryArray.from_buffers(
            pa.binary(byte_width),
            len(fossils),
            [None, pa.py_buffer(fossils)],
        )
        gc.collect()

        log(" - creating DataFrame")
        df = pl.DataFrame(
            {
                "data_raw": pl.Series(fossils, dtype=pl.Binary),
            },
        )
        del fossils
        gc.collect()

        log(f" - data_raw: {df['data_raw'].head(3)}")
        assert (df["data_raw"].bin.size(unit="b") == byte_width).all()

        len_df = len(df)
        nPos = nCol * nRow
        log(f" -{nPos=} {len_df=}...")

        tmp_from_pqt, tmp_to_pqt = "tmpfossils-from.pqt", "tmpfossils-to.pqt"
        log(f"writing to {tmp_from_pqt}...")
        df.write_parquet(tmp_from_pqt, compression="lz4")
        log("... done!")

        log("clearing memory...")
        del df
        gc.collect()
        log("... done!")

        log(" - calculating indices...")  # use numpy to save memory vs polars
        layer = np.arange(len_df, dtype=np.uint32) // nPos
        layer_T = np.array(layer_T, dtype=np.uint64)[layer]
        position = np.arange(len_df, dtype=np.uint32) % nPos

        log(" - creating indices df...")  # use numpy to save memory vs polars
        df_indices = pl.DataFrame(
            {
                "layer": pl.Series(layer, dtype=pl.UInt32),
                "layer_T": pl.Series(layer_T, dtype=pl.UInt64),
                "position": pl.Series(position, dtype=pl.UInt32),
            },
        ).lazy()
        del layer, layer_T, position
        gc.collect()

        log(" - saving indices...")  # separate for sink compat
        fi_path = (
            "a=fossil-indices"
            f"+flavor={genomeFlavor}"
            f"+seed={globalSeed}"
            f"+ncycle={nCycleAtLeast}"
            "+ext=.pqt"
        )
        write_parquet_verbose(df_indices, fi_path)
        del df_indices
        gc.collect()

        log(f"scanning {tmp_from_pqt}...")
        df = pl.scan_parquet(tmp_from_pqt, cache=False)
        gc.collect()

        log(f" - encoding binary fossil rows to hex...")
        df = df.with_columns(
            data_hex=pl.col("data_raw").bin.encode("hex"),
        ).drop("data_raw")

        log(
            f""" - data_hex: {
                df.select('data_hex').head(3).collect().to_series()
            }""",
        )

        log(f"sinking to {tmp_to_pqt}...")
        df.sink_parquet(tmp_to_pqt, compression="lz4")
        log("... done!")

        log(f"moving {tmp_to_pqt} to {tmp_from_pqt}...")
        shutil.move(tmp_to_pqt, tmp_from_pqt)
        log("... done!")

        log(f"scanning {tmp_from_pqt}...")
        df = pl.scan_parquet(tmp_from_pqt, cache=False)
        gc.collect()

        log(" - validation check...")
        validation_exprs = [
            pl.col("data_hex").str.len_chars() == byte_width * 2,
            pl.col("data_hex").str.len_bytes() == byte_width * 2,
            pl.col("data_hex").str.contains("^[0-9a-fA-F]+$"),
            pl.col("data_hex").str.head(8) == pl.col("data_hex").str.tail(8),
        ]
        validation_result = all(
            df.filter(~validation_expr).collect().is_empty()
            for validation_expr in tqdm(validation_exprs, desc="validation")
        )
        log(f" - {validation_result=}")
        if not validation_result:
            for i, expr in enumerate(validation_exprs):
                nfail = df.filter(~expr).select(pl.len()).collect().item()
                log(f"  - validation  {i=} {str(expr)=} failed {nfail=} rows")
                if nfail:
                    log(
                        f"""  - example failed data_hex: {
                            df.filter(~expr).select('data_hex')
                            .head(3).collect().to_series()
                        }""",
                    )
        assert validation_result

        log(" - stripping bookends...")
        df = df.with_columns(pl.col("data_hex").str.head(-8).str.tail(-8))

        log(
            f""" - data_hex: {
                df.select('data_hex').head(3).collect().to_series()
            }""",
        )

        log(f"sinking to {tmp_to_pqt}...")
        df.sink_parquet(tmp_to_pqt, compression="lz4")
        log("... done!")

        log(f"moving {tmp_to_pqt} to {tmp_from_pqt}...")
        shutil.move(tmp_to_pqt, tmp_from_pqt)
        log("... done!")

        log(f"scanning {tmp_from_pqt}...")
        df = pl.scan_parquet(tmp_from_pqt, cache=False)
        gc.collect()

        log("- adding metadata columns")
        df = df.with_columns(
            [
                pl.lit(value, dtype=dtype).alias(key)
                for key, (value, dtype) in metadata.items()
            ],
            is_extant=False,
        )

        fg_path = (
            "a=fossil-genomes"
            f"+flavor={genomeFlavor}"
            f"+seed={globalSeed}"
            f"+ncycle={nCycleAtLeast}"
            "+ext=.pqt"
        )
        write_parquet_verbose(df, fg_path)

        log("- cleaning up...")
        del df
        gc.collect()
        pathlib.Path(tmp_from_pqt).unlink(missing_ok=True)
        pathlib.Path(tmp_to_pqt).unlink(missing_ok=True)
        log("... done!")

        log("- concatenating fossil genomes and indices...")
        df = pl.concat(
            [
                pl.scan_parquet(fg_path, cache=False),
                pl.scan_parquet(fi_path, cache=False),
            ],
            how="horizontal",
        )
        log(f" - {df=}")
        for how in pl.LazyFrame.lazy, pl.LazyFrame.collect:
            log(f" - trying {how=}")
            try:
                write_parquet_verbose(
                    how(df),
                    "a=fossils"
                    f"+flavor={genomeFlavor}"
                    f"+seed={globalSeed}"
                    f"+ncycle={nCycleAtLeast}"
                    "+ext=.pqt",
                )
            except pl.exceptions.InvalidOperationError as e:
                log(f" - {how=} {str(e)}")
            else:
                log(f" - {how=} success")
                break
        else:
            assert False, "fossil write failed"

        log("- ... done!")

    else:
        log("- no fossils to process!")


log("- printenv")
for k, v in sorted(os.environ.items()):
    log(f"  - {k}={v}")

log("- setting up temp dir")
# need to add polars to Cerebras python
temp_dir = f"{os.getenv('ASYNC_GA_LOCAL_PATH', 'local')}/tmp/{uuid.uuid4()}"
os.makedirs(temp_dir, exist_ok=True)
atexit.register(shutil.rmtree, temp_dir, ignore_errors=True)
log(f"  - {temp_dir=}")

log("- installing downstream")
for attempt in range(4):
    try:
        subprocess.check_call(
            [
                "pip",
                "install",
                f"--target={temp_dir}",
                "--no-cache-dir",
                "--no-deps",  # prevent numpy from being reinstalled
                "--ignore-requires-python",  # some components require py3.10+
                "downstream==1.16.4",
                "lazy_loader==0.4",
            ],
            env={
                **os.environ,
                "TMPDIR": temp_dir,
            },
        )
        log("- pip install succeeded!")
        break
    except subprocess.CalledProcessError as e:
        log(e)
        log(f"retrying {attempt=}...")
else:
    raise e
log(f"- extending sys path with temp dir {temp_dir=}")
sys.path.append(temp_dir)

import downstream  # type: ignore
from downstream import dstream  # type: ignore

log(f"- downstream version: {downstream.__version__}")

try:
    import polars  # type: ignore

    log("- polars already installed, skipping installation")
    del polars
except ImportError:
    log("- installing polars")
    for attempt in range(4):
        try:
            subprocess.check_call(
                [
                    "pip",
                    "install",
                    f"--target={temp_dir}",
                    "--no-cache-dir",
                    "polars==1.8.2",
                ],
                env={
                    **os.environ,
                    "TMPDIR": temp_dir,
                },
            )
            log("- pip install succeeded!")
            break
        except subprocess.CalledProcessError as e:
            log(e)
            log(f"retrying {attempt=}...")
    else:
        raise e
    log(f"- extending sys path with temp dir {temp_dir=}")
    sys.path.append(temp_dir)


log("- importing third-party dependencies")
log("  - numpy...")
import numpy as np

log("  - polars...")
import polars as pl

log("  - pyarrow...")
import pyarrow as pa

log("  - scipy...")
from scipy import stats as sps

log("  - tqdm...")
from tqdm import tqdm

log("- defining helper functions")


def write_parquet_verbose(df: pl.DataFrame, file_name: str) -> None:
    log(f"saving df to {file_name=}")
    if isinstance(df, pl.DataFrame):
        log(f"- {df.shape=}")
    else:
        log(f" - {type(df)} {df.collect_schema().len()=}")

    tmp_file = f"{os.getenv('ASYNC_GA_LOCAL_PATH', 'local')}/tmp.pqt"
    os.makedirs(os.path.dirname(os.path.abspath(tmp_file)), exist_ok=True)
    if isinstance(df, pl.DataFrame):
        df.write_parquet(tmp_file, compression="lz4")
    else:
        df.sink_parquet(tmp_file, compression="lz4")
    log("- write_parquet complete")

    file_size_mb = os.path.getsize(tmp_file) / (1024 * 1024)
    log(f"- saved file size: {file_size_mb:.2f} MB")

    lazy_frame = pl.scan_parquet(tmp_file)
    if file_size_mb <= 1024:
        log("- LazyFrame describe:")
        log(lazy_frame.describe())
    else:
        log("- LazyFrame describe skipped due to large file size")

    original_row_count = df.lazy().select(pl.len()).collect().item()
    lazy_row_count = lazy_frame.select(pl.len()).collect().item()
    assert lazy_row_count == original_row_count, (
        f"Row count mismatch between original and lazy frames: "
        f"{original_row_count=}, {lazy_row_count=}"
    )

    os.makedirs(os.path.dirname(os.path.abspath(file_name)), exist_ok=True)
    shutil.move(tmp_file, file_name)
    log(f"- move {tmp_file} to destination {file_name} complete")

    log("- verbose save complete!")


# adapted from https://stackoverflow.com/a/31347222/17332200
def add_bool_arg(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--" + name, dest=name.replace("-", "_"), action="store_true"
    )
    group.add_argument(
        "--no-" + name, dest=name.replace("-", "_"), action="store_false"
    )
    parser.set_defaults(**{name.replace("-", "_"): default})


log("- reading env variables")
# number of rows, columns, and genome words
nCol = int(os.getenv("ASYNC_GA_NCOL", 3))
nRow = int(os.getenv("ASYNC_GA_NROW", 3))
nWav = int(os.getenv("ASYNC_GA_NWAV", -1))
nTrait = int(os.getenv("ASYNC_GA_NTRAIT", 1))
log(f"{nCol=}, {nRow=}, {nWav=}, {nTrait=}")

log("- setting global variables")
wavSize = 32  # number of bits in a wavelet
tscSizeWords = 3  # number of 16-bit values in 48-bit timestamp values
tscSizeWords += tscSizeWords % 2  # make even multiple of 32-bit words
tscTicksPerSecond = 850 * 10**6  # 850 MHz

log("- configuring argparse")
parser = argparse.ArgumentParser()
parser.add_argument("--name", help="the test compile output dir", default="out")
add_bool_arg(parser, "suptrace", default=True)
add_bool_arg(parser, "process-fossils", default=None)
parser.add_argument("--cmaddr", help="IP:port for CS system")
log("- parsing arguments")
args = parser.parse_args()

log("args =======================================================")
log(args)

log("metadata ===================================================")
with open(f"{args.name}/out.json", encoding="utf-8") as json_file:
    compile_data = json.load(json_file)

globalSeed = int(compile_data["params"]["globalSeed"])
nCycleAtLeast = int(compile_data["params"]["nCycleAtLeast"])
msecAtLeast = int(compile_data["params"]["msecAtLeast"])
tscAtLeast = int(compile_data["params"]["tscAtLeast"])
nColSubgrid = int(compile_data["params"]["nColSubgrid"])
nRowSubgrid = int(compile_data["params"]["nRowSubgrid"])
nonBlock = bool(int(compile_data["params"]["nonBlock"]))
tilePopSize = int(compile_data["params"]["popSize"])
tournSize = (
    float(compile_data["params"]["tournSizeNumerator"])
    / float(compile_data["params"]["tournSizeDenominator"])
)

with open("compconf.json", encoding="utf-8") as json_file:
    compconf_data = json.load(json_file)

log(f" - applying globalSeed={globalSeed}")
random.seed(globalSeed)
np.random.seed(globalSeed)

log(f" - {compconf_data=}")

traitLoggerNumBits = int(compconf_data["CEREBRASLIB_TRAITLOGGER_NUM_BITS:u32"])
assert bin(traitLoggerNumBits)[2:].count("1") == 1
traitLoggerDstreamAlgoName = compconf_data[
    "CEREBRASLIB_TRAITLOGGER_DSTREAM_ALGO_NAME:comptime_string"
]
log(f" - {traitLoggerNumBits=} {traitLoggerDstreamAlgoName=}")

genomeFlavor = compconf_data["ASYNC_GA_GENOME_FLAVOR:comptime_string"]
log(f" - {genomeFlavor=}")
genomePath = f"{os.getenv('ASYNC_GA_CEREBRASLIB_PATH', 'cerebraslib')}/genome/{genomeFlavor}.csl"
log(f" - reading genome data from {genomePath}")
genomeDataRaw = "".join(
    removeprefix(line, "//!").strip()
    for line in pathlib.Path(genomePath).read_text().split("\n")
    if line.startswith("//!")
) or "{}"
genomeData = eval(genomeDataRaw, {"compconf_data": compconf_data, "pl": pl})
log(f" - {genomeData=}")

assert nWav in (genomeData["nWav"][0], -1)
nWav = genomeData["nWav"][0]

metadata = {
    "genomeFlavor": (genomeFlavor, pl.Categorical),
    "globalSeed": (globalSeed, pl.UInt32),
    "nCol": (nCol, pl.UInt16),
    "nRow": (nRow, pl.UInt16),
    "nWav": (nWav, pl.UInt8),
    "nTrait": (nTrait, pl.UInt8),
    "nCycle": (nCycleAtLeast, pl.UInt32),
    "nColSubgrid": (nColSubgrid, pl.UInt16),
    "nRowSubgrid": (nRowSubgrid, pl.UInt16),
    "nonBlock": (nonBlock, pl.Boolean),
    "tilePopSize": (tilePopSize, pl.UInt16),
    "tournSize": (tournSize, pl.Float32),
    "msec": (msecAtLeast, pl.Float32),
    "tsc": (tscAtLeast, pl.UInt64),
    "replicate": (str(uuid.uuid4()), pl.Categorical),
    **genomeData,
    **{
        k.split(":")[0]: {
            "bool": lambda: (json.loads(v), pl.Boolean),
            "f16": lambda: (float(v), pl.Float32),
            "f32": lambda: (float(v), pl.Float32),
            "i8": lambda: (int(v), pl.Int8),
            "i16": lambda: (int(v), pl.Int16),
            "i32": lambda: (int(v), pl.Int32),
            "i64": lambda: (int(v), pl.Int64),
            "u8": lambda: (int(v), pl.UInt8),
            "u16": lambda: (int(v), pl.UInt16),
            "u32": lambda: (int(v), pl.UInt32),
            "u64": lambda: (int(v), pl.UInt64),
            "comptime_string": lambda: (v, pl.Categorical),
        }[k.split(":")[-1]]()
        for k, v in compconf_data.items()
    },
}
log(metadata)

if args.process_fossils is True:
    log(f" - processing fossils {nWav=}...")
    process_fossils(nWav)
    log(" - done! exiting...")
    sys.exit(0)

log("- setting up fossil storage")

max_fossil_sets = int(os.getenv("ASYNC_GA_MAX_FOSSIL_SETS", 2**32 - 1))
log(f" - {max_fossil_sets=}")

max_fossil_sets_dstream_algo = os.getenv(
    "ASYNC_GA_MAX_FOSSIL_SETS_DSTREAM_ALGO", "dstream.sticky_algo"
)
log(f" - {max_fossil_sets_dstream_algo=}")

max_fossil_sets_dstream_algo = eval(
    max_fossil_sets_dstream_algo, {"dstream": dstream}
)
log(f" - {max_fossil_sets_dstream_algo=}")

fossils = set()
fossil_mmap = np.memmap(
    f"{temp_dir}/fossils.dat",
    dtype=np.uint32,
    mode="w+",
    shape=(max_fossil_sets, nCol, nRow, nWav + 2),
)

log("- importing cerebras depencencies")
from cerebras.sdk.runtime.sdkruntimepybind import (
    MemcpyDataType,
    MemcpyOrder,
    SdkRuntime,
)  # pylint: disable=no-name-in-module

log("do run =====================================================")
# Path to ELF and simulation output files
runner = SdkRuntime(
    "out", cmaddr=args.cmaddr, suppress_simfab_trace=args.suptrace
)
log("- SdkRuntime created")

runner.load()
log("- runner loaded")

runner.run()
log("- runner run ran")

runner.launch("dolaunch", nonblock=False)
launch_ns = time.time_ns()
log(f"- runner launch complete {launch_ns=}")

log(f"- {nonBlock=}, if True waiting for first kernel to finish...")
for cycle, __ in enumerate(it.takewhile(bool, it.repeat(nonBlock))):
    elapsed_ns = time.time_ns() - launch_ns
    log_cycle = elapsed_ns // (10**9 * 20)
    if log_once(f"\n - {20 * log_cycle} seconds elapsed"):
        log(f" ! phase=1 {log_cycle=} {cycle=} {len(fossils)=} {elapsed_ns=}")
        print(flush=True)

    print("1", end="", flush=True)

    print(f"({len(fossils)})", end="", flush=True)
    print("a", end="", flush=True)
    memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
    out_tensors = np.zeros((nCol, nRow, nWav + 2), np.uint32)
    print("b", end="", flush=True)
    runner.memcpy_d2h(
        out_tensors.ravel(),
        runner.get_id("genomeBookend"),
        0,  # x0
        0,  # y0
        nCol,  # width
        nRow,  # height
        nWav + 2,  # num wavelets
        streaming=False,
        data_type=memcpy_dtype,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )
    print("c", end="", flush=True)

    fossil_storage_site = max_fossil_sets_dstream_algo.assign_storage_site(
        S=max_fossil_sets,
        T=cycle,
    )
    print("d", end="", flush=True)

    if fossil_storage_site is not None:
        genome_data = out_tensors.copy()
        print("e", end="", flush=True)
        fossil_mmap[fossil_storage_site, :, :, :] = genome_data
        print("f", end="", flush=True)
        fossils.add(fossil_storage_site)
        print("g", end="", flush=True)
    else:
        print("h", end="", flush=True)

    print("2", end="", flush=True)
    memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
    out_tensors = np.zeros((nCol, nRow, 1), np.uint32)
    runner.memcpy_d2h(
        out_tensors.ravel(),
        runner.get_id("cycleCounter"),
        0,  # x0
        0,  # y0
        nCol,  # width
        nRow,  # height
        1,  # num wavelets
        streaming=False,
        data_type=memcpy_dtype,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )
    print("3", end="", flush=True)

    cycle_counts = out_tensors.ravel().copy()
    num_complete = np.sum(cycle_counts >= nCycleAtLeast)
    print("4", end="", flush=True)

    should_break = num_complete > 0
    print(f"({num_complete/cycle_counts.size * 100}%)", end="", flush=True)
    if should_break:
        phase1_elapsed_ns = time.time_ns() - launch_ns
        phase1_elapsed_cycles = cycle + 1
        print("!", flush=True)
        break
    else:
        print("5", end="", flush=True)
        runner.launch("dorefresh", nonblock=False)
        print("|", end="", flush=True)
        continue

log(f"- {nonBlock=}, if True waiting for last kernel to finish...")
for cycle, __ in enumerate(it.takewhile(bool, it.repeat(nonBlock))):
    elapsed_ns = time.time_ns() - launch_ns
    log_cycle = elapsed_ns // (10**9 * 20)
    if log_once(f"\n - {20 * log_cycle} seconds elapsed"):
        log(f" ! phase=2 {log_cycle=} {cycle=} {len(fossils)=} {elapsed_ns=}")
        print(flush=True)

    print("1", end="", flush=True)
    memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
    out_tensors = np.zeros((nCol, nRow, 1), np.uint32)
    runner.memcpy_d2h(
        out_tensors.ravel(),
        runner.get_id("cycleCounter"),
        0,  # x0
        0,  # y0
        nCol,  # width
        nRow,  # height
        1,  # num wavelets
        streaming=False,
        data_type=memcpy_dtype,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )
    print("2", end="", flush=True)

    cycle_counts = out_tensors.ravel().copy()
    num_complete = np.sum(cycle_counts >= nCycleAtLeast)
    print("3", end="", flush=True)
    should_break = num_complete == cycle_counts.size
    print(f"({num_complete/cycle_counts.size * 100}%)", end="", flush=True)
    if should_break:
        phase2_elapsed_ns = time.time_ns() - launch_ns
        phase2_elapsed_cycles = cycle + 1
        print("!", flush=True)
        break
    else:
        print("|", end="", flush=True)
        continue

log("- run complete!")
log(f" - run elapsed seconds: {(time.time_ns() - launch_ns) / (10 ** 9):.3f}")
if nonBlock:
    total_elapsed_ns = phase1_elapsed_ns + phase2_elapsed_ns
    log(f" - {phase1_elapsed_ns=}")
    log(f" - {phase1_elapsed_ns / total_elapsed_ns=}")
    log(f" - {phase2_elapsed_ns=}")
    log(f" - {phase2_elapsed_ns / total_elapsed_ns=}")
    log(f" - {total_elapsed_ns=}")

    total_elapsed_cycles = phase1_elapsed_cycles + phase2_elapsed_cycles
    log(f"- {phase1_elapsed_cycles=}")
    log(f"- {phase1_elapsed_cycles / total_elapsed_cycles=}")
    log(f"- {phase2_elapsed_cycles=}")
    log(f"- {phase2_elapsed_cycles / total_elapsed_cycles=}")
    log(f"- {total_elapsed_cycles=}")

    log("arranging fossils ==================================================")
    log(f" - {len(fossils)=}")
    assert len(fossils) <= max_fossil_sets

    max_words = max(fossils, default=0) + 1
    log(f" - {max_words=}")
    fossil_layer_T = [
        *it.islice(
            max_fossil_sets_dstream_algo.lookup_ingest_times(
                S=max_fossil_sets,
                T=phase1_elapsed_cycles,
            ),
            max_words,
        )
    ]
    fossils = sorted(fossils, key=fossil_layer_T.__getitem__)

log("thinning fossils =======================================================")
log(f" - {len(fossils)=}")
assert len(fossils) <= max_fossil_sets

max_fossil_sets_spread = int(
    os.getenv("ASYNC_GA_MAX_FOSSIL_SETS_SPREAD", 2**32 - 1)
)
log(f" - {max_fossil_sets_spread=}")
m = min(max_fossil_sets_spread, len(fossils))
if m < len(fossils):
    log(f" - spacing to {m} fossil sets...")
    # adapted from https://stackoverflow.com/a/9873804
    fossils = [
        fossils[i * len(fossils) // m + len(fossils) // (2 * m)]
        for i in range(m)
    ]
    log(f" - {len(fossils)=}")

max_fossil_sets_sample = int(
    os.getenv("ASYNC_GA_MAX_FOSSIL_SETS_SAMPLE", 2**32 - 1)
)
log(f" - {max_fossil_sets_sample=}")
m = min(max_fossil_sets_sample, len(fossils))
if m < len(fossils):
    log(f" - sampling {m} fossil sets...")
    fossils = [
        fossils[i] for i in sorted(random.sample(range(len(fossils)), k=m))
    ]
    log(f" - {len(fossils)=}")

log("writing fossils ========================================================")
os.makedirs("raw", exist_ok=True)
np.savez(
    "raw/fossils.npz",
    **{
        str(fossil_layer_T[fossil_index]): fossil_mmap[fossil_index, :, :, :]
        for fossil_index in fossils
    },
)
log("- done!")
file_size_gb = os.path.getsize("raw/fossils.npz") / (1024 * 1024 * 1024)
log(f"- saved file size: {file_size_gb:.2f} GB")

del fossils

log("processing fossils =====================================================")
if args.process_fossils is False:
    log(" - skipping fossil processing!")
else:
    log(f" - processing fossils {nWav=}...")
    process_fossils(nWav)

log("whoami =====================================================")
memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
out_tensors = np.zeros((nCol, nRow), np.uint32)

runner.memcpy_d2h(
    out_tensors.ravel(),
    runner.get_id("whoami"),
    0,  # x0
    0,  # y0
    nCol,  # width
    nRow,  # height
    1,  # num wavelets
    streaming=False,
    data_type=memcpy_dtype,
    order=MemcpyOrder.ROW_MAJOR,
    nonblock=False,
)
whoami_data = out_tensors.copy()
log(whoami_data[:20, :20])

log("whereami x =================================================")
memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
out_tensors = np.zeros((nCol, nRow), np.uint32)

runner.memcpy_d2h(
    out_tensors.ravel(),
    runner.get_id("whereami_x"),
    0,  # x0
    0,  # y0
    nCol,  # width
    nRow,  # height
    1,  # num wavelets
    streaming=False,
    data_type=memcpy_dtype,
    order=MemcpyOrder.ROW_MAJOR,
    nonblock=False,
)
whereami_x_data = out_tensors.copy()
log(whereami_x_data[:20, :20])

log("whereami y =================================================")
memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
out_tensors = np.zeros((nCol, nRow), np.uint32)

runner.memcpy_d2h(
    out_tensors.ravel(),
    runner.get_id("whereami_y"),
    0,  # x0
    0,  # y0
    nCol,  # width
    nRow,  # height
    1,  # num wavelets
    streaming=False,
    data_type=memcpy_dtype,
    order=MemcpyOrder.ROW_MAJOR,
    nonblock=False,
)
whereami_y_data = out_tensors.copy()
log(whereami_y_data[:20, :20])

log("trait data =================================================")
memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
out_tensors = np.zeros((nCol, nRow, nTrait), np.uint32)
runner.memcpy_d2h(
    out_tensors.ravel(),
    runner.get_id("traitCounts"),
    0,  # x0
    0,  # y0
    nCol,  # width
    nRow,  # height
    nTrait,  # num possible trait values
    streaming=False,
    data_type=memcpy_dtype,
    order=MemcpyOrder.ROW_MAJOR,
    nonblock=False,
)
traitCounts_data = out_tensors.copy()
log(f"traitCounts_data {Counter(traitCounts_data.ravel())}")

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
out_tensors = np.zeros((nCol, nRow, nTrait), np.uint32)
runner.memcpy_d2h(
    out_tensors.ravel(),
    runner.get_id("traitCycles"),
    0,  # x0
    0,  # y0
    nCol,  # width
    nRow,  # height
    nTrait,  # num possible trait values
    streaming=False,
    data_type=memcpy_dtype,
    order=MemcpyOrder.ROW_MAJOR,
    nonblock=False,
)
traitCycles_data = out_tensors.copy()
log(f"traitCycles_data {Counter(traitCycles_data.ravel())}")

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
out_tensors = np.zeros((nCol, nRow, nTrait), np.uint32)
runner.memcpy_d2h(
    out_tensors.ravel(),
    runner.get_id("traitValues"),
    0,  # x0
    0,  # y0
    nCol,  # width
    nRow,  # height
    nTrait,  # num possible trait values
    streaming=False,
    data_type=memcpy_dtype,
    order=MemcpyOrder.ROW_MAJOR,
    nonblock=False,
)
traitValues_data = out_tensors.copy()
log(f"traitValues_data {str(Counter(traitValues_data.ravel()))[:500]}")

# save trait data values to a file
df = pl.DataFrame({
    "trait count": pl.Series(traitCounts_data.ravel(), dtype=pl.UInt16),
    "trait cycle last seen": pl.Series(traitCycles_data.ravel(), dtype=pl.UInt32),
    "trait value": pl.Series(traitValues_data.ravel(), dtype=pl.UInt8),
    "tile": pl.Series(np.repeat(whoami_data.ravel(), nTrait), dtype=pl.UInt32),
    "row": pl.Series(np.repeat(whereami_y_data.ravel(), nTrait), dtype=pl.UInt16),
    "col": pl.Series(np.repeat(whereami_x_data.ravel(), nTrait), dtype=pl.UInt16),
}).with_columns([
    pl.lit(value, dtype=dtype).alias(key)
    for key, (value, dtype) in metadata.items()
])


for trait, group in df.group_by("trait value"):
    log(f"trait {trait} total count is {group['trait count'].sum()}")

write_parquet_verbose(
    df,
    "a=traits"
    f"+flavor={genomeFlavor}"
    f"+seed={globalSeed}"
    f"+ncycle={nCycleAtLeast}"
    "+ext=.pqt",
)
del df, traitCounts_data, traitCycles_data, traitValues_data

log("wildtype traitlogs ==============================================")
memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
traitLoggerNumWavs = traitLoggerNumBits // wavSize + 1  # +1 for dstream_T
out_tensors = np.zeros((nCol, nRow, traitLoggerNumWavs), np.uint32)

runner.memcpy_d2h(
    out_tensors.ravel(),
    runner.get_id("wildtypeLoggerRecord"),
    0,  # x0
    0,  # y0
    nCol,  # width
    nRow,  # height
    traitLoggerNumWavs,  # num elements
    streaming=False,
    data_type=memcpy_dtype,
    order=MemcpyOrder.ROW_MAJOR,
    nonblock=False,
)
raw_binary_data = out_tensors.copy()
log("entering assemble_binary_data from wildtype traitlogs...")
record_raw = assemble_binary_data(
    raw_binary_data.view(np.uint32), nWav=traitLoggerNumWavs, verbose=True
)
log(f" - casting record_raw to object")
record_raw = record_raw.astype(object)

# save trait logger values to a file
log(" - creating DataFrame")
df = pl.DataFrame({
    "data_raw": pl.Series(record_raw, dtype=pl.Binary),
    "tile": pl.Series(whoami_data.ravel(), dtype=pl.UInt32),
    "row": pl.Series(whereami_y_data.ravel(), dtype=pl.UInt16),
    "col": pl.Series(whereami_x_data.ravel(), dtype=pl.UInt16),
}).with_columns(
    pl.lit(value, dtype=dtype).alias(key)
    for key, (value, dtype) in metadata.items()
)
log(f" - data_raw: {df['data_raw'].head(3)}")
assert (df["data_raw"].bin.size(unit="b") == traitLoggerNumWavs * 4).all()
df = df.with_columns(
    data_hex=pl.col("data_raw").bin.encode("hex"),
    dstream_algo=pl.lit(
        f"dstream.{traitLoggerDstreamAlgoName}", dtype=pl.Categorical
    ),
    dstream_storage_bitoffset=pl.lit(0, dtype=pl.UInt16),
    dstream_storage_bitwidth=pl.lit(traitLoggerNumBits, dtype=pl.UInt16),
    dstream_S=pl.lit(traitLoggerNumBits, dtype=pl.UInt16),
    dstream_T_bitoffset=pl.lit(traitLoggerNumBits, dtype=pl.UInt16),
    dstream_T_bitwidth=pl.lit(32, dtype=pl.UInt16),
    trait_value=pl.lit(0, dtype=pl.UInt16),
).drop("data_raw")

log(f" - data_hex: {df['data_hex'].head(3)}")
assert (df["data_hex"].str.len_chars() == traitLoggerNumWavs * 8).all()
assert (df["data_hex"].str.len_bytes() == traitLoggerNumWavs * 8).all()
assert (df["data_hex"].str.contains("^[0-9a-fA-F]+$")).all()

write_parquet_verbose(
    df,
    "a=traitloggerRecord"
    f"+flavor={genomeFlavor}"
    f"+seed={globalSeed}"
    f"+ncycle={nCycleAtLeast}"
    "+ext=.pqt",
)
del df, raw_binary_data, record_raw

log("fitness ===================================================")
memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
out_tensors = np.zeros((nCol, nRow), np.float32)

runner.memcpy_d2h(
    out_tensors.ravel(),
    runner.get_id("fitness"),
    0,  # x0
    0,  # y0
    nCol,  # width
    nRow,  # height
    1,  # num wavelets
    streaming=False,
    data_type=memcpy_dtype,
    order=MemcpyOrder.ROW_MAJOR,
    nonblock=False,
)
fitness_data = out_tensors.copy()
log(fitness_data[:20, :20])

log("genome values ==============================================")
memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
out_tensors = np.zeros((nCol, nRow, nWav), np.uint32)

runner.memcpy_d2h(
    out_tensors.ravel(),
    runner.get_id("genome"),
    0,  # x0
    0,  # y0
    nCol,  # width
    nRow,  # height
    nWav,  # num wavelets
    streaming=False,
    data_type=memcpy_dtype,
    order=MemcpyOrder.ROW_MAJOR,
    nonblock=False,
)
raw_genome_data = out_tensors.copy()
genome_raw = assemble_genome_data(raw_genome_data, verbose=True)
log(f" - casting genome_raw to object")
genome_raw = genome_raw.astype(object)

# save genome values to a file
log(" - creating DataFrame")
df = pl.DataFrame({
    "data_raw": pl.Series(genome_raw, dtype=pl.Binary),
    "is_extant": True,
    "fitness": pl.Series(fitness_data.ravel(), dtype=pl.Float32),
    "tile": pl.Series(whoami_data.ravel(), dtype=pl.UInt32),
    "row": pl.Series(whereami_y_data.ravel(), dtype=pl.UInt16),
    "col": pl.Series(whereami_x_data.ravel(), dtype=pl.UInt16),
})
log(f" - data_raw: {df['data_raw'].head(3)}")
assert (df["data_raw"].bin.size(unit="b") == nWav * 4).all()
df = df.with_columns(
    *(
        pl.lit(value, dtype=dtype).alias(key)
        for key, (value, dtype) in metadata.items()
    ),
    data_hex=pl.col("data_raw").bin.encode("hex"),
).drop("data_raw")

log(f" - data_hex: {df['data_hex'].head(3)}")
assert (df["data_hex"].str.len_chars() == nWav * 8).all()
assert (df["data_hex"].str.len_bytes() == nWav * 8).all()
assert (df["data_hex"].str.contains("^[0-9a-fA-F]+$")).all()

write_parquet_verbose(
    df,
    "a=genomes"
    f"+flavor={genomeFlavor}"
    f"+seed={globalSeed}"
    f"+ncycle={nCycleAtLeast}"
    "+ext=.pqt",
)
del df, fitness_data, genome_raw

log("cycle counter =============================================")
memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
out_tensors = np.zeros((nCol, nRow), np.uint32)

runner.memcpy_d2h(
    out_tensors.ravel(),
    runner.get_id("cycleCounter"),
    0,  # x0
    0,  # y0
    nCol,  # width
    nRow,  # height
    1,  # num wavelets
    streaming=False,
    data_type=memcpy_dtype,
    order=MemcpyOrder.ROW_MAJOR,
    nonblock=False,
)
cycle_counts = out_tensors.ravel().copy()
log(cycle_counts[:100])


log("recv counter N ==============================================")
memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
out_tensors = np.zeros((nCol, nRow), np.uint32)

runner.memcpy_d2h(
    out_tensors.ravel(),
    runner.get_id("recvCounter_N"),
    0,  # x0
    0,  # y0
    nCol,  # width
    nRow,  # height
    1,  # num wavelets
    streaming=False,
    data_type=memcpy_dtype,
    order=MemcpyOrder.ROW_MAJOR,
    nonblock=False,
)
recvN = out_tensors.copy()
log(recvN[:20, :20])

log("recv counter S ==============================================")
memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
out_tensors = np.zeros((nCol, nRow), np.uint32)

runner.memcpy_d2h(
    out_tensors.ravel(),
    runner.get_id("recvCounter_S"),
    0,  # x0
    0,  # y0
    nCol,  # width
    nRow,  # height
    1,  # num wavelets
    streaming=False,
    data_type=memcpy_dtype,
    order=MemcpyOrder.ROW_MAJOR,
    nonblock=False,
)
recvS = out_tensors.copy()
log(recvS[:20, :20])

log("recv counter E ==============================================")
memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
out_tensors = np.zeros((nCol, nRow), np.uint32)

runner.memcpy_d2h(
    out_tensors.ravel(),
    runner.get_id("recvCounter_E"),
    0,  # x0
    0,  # y0
    nCol,  # width
    nRow,  # height
    1,  # num wavelets
    streaming=False,
    data_type=memcpy_dtype,
    order=MemcpyOrder.ROW_MAJOR,
    nonblock=False,
)
recvE = out_tensors.copy()
log(recvE[:20, :20])

log("recv counter W ==============================================")
memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
out_tensors = np.zeros((nCol, nRow), np.uint32)

runner.memcpy_d2h(
    out_tensors.ravel(),
    runner.get_id("recvCounter_W"),
    0,  # x0
    0,  # y0
    nCol,  # width
    nRow,  # height
    1,  # num wavelets
    streaming=False,
    data_type=memcpy_dtype,
    order=MemcpyOrder.ROW_MAJOR,
    nonblock=False,
)
recvW = out_tensors.copy()
log(recvW[:20, :20])

log("recv counter sum ===========================================")
recvSum = [
    *map(sum, zip(recvN.ravel(), recvS.ravel(), recvE.ravel(), recvW.ravel()))
]
log(recvSum[:100])
log(f"{np.mean(recvSum)=} {np.std(recvSum)=} {sps.sem(recvSum)=}")
log(f"{np.median(recvSum)=} {np.min(recvSum)=} {np.max(recvSum)=}")

log("send counter N ==============================================")
memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
out_tensors = np.zeros((nCol, nRow), np.uint32)

runner.memcpy_d2h(
    out_tensors.ravel(),
    runner.get_id("sendCounter_N"),
    0,  # x0
    0,  # y0
    nCol,  # width
    nRow,  # height
    1,  # num wavelets
    streaming=False,
    data_type=memcpy_dtype,
    order=MemcpyOrder.ROW_MAJOR,
    nonblock=False,
)
sendN = out_tensors.copy()
log(sendN[:20, :20])

log("send counter S ==============================================")
memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
out_tensors = np.zeros((nCol, nRow), np.uint32)

runner.memcpy_d2h(
    out_tensors.ravel(),
    runner.get_id("sendCounter_S"),
    0,  # x0
    0,  # y0
    nCol,  # width
    nRow,  # height
    1,  # num wavelets
    streaming=False,
    data_type=memcpy_dtype,
    order=MemcpyOrder.ROW_MAJOR,
    nonblock=False,
)
sendS = out_tensors.copy()
log(sendS[:20, :20])

log("send counter E ==============================================")
memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
out_tensors = np.zeros((nCol, nRow), np.uint32)

runner.memcpy_d2h(
    out_tensors.ravel(),
    runner.get_id("sendCounter_E"),
    0,  # x0
    0,  # y0
    nCol,  # width
    nRow,  # height
    1,  # num wavelets
    streaming=False,
    data_type=memcpy_dtype,
    order=MemcpyOrder.ROW_MAJOR,
    nonblock=False,
)
sendE = out_tensors.copy()
log(sendE[:20, :20])

log("send counter W ==============================================")
memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
out_tensors = np.zeros((nCol, nRow), np.uint32)

runner.memcpy_d2h(
    out_tensors.ravel(),
    runner.get_id("sendCounter_W"),
    0,  # x0
    0,  # y0
    nCol,  # width
    nRow,  # height
    1,  # num wavelets
    streaming=False,
    data_type=memcpy_dtype,
    order=MemcpyOrder.ROW_MAJOR,
    nonblock=False,
)
sendW = out_tensors.copy()
log(sendW[:20, :20])

log("send counter sum ===========================================")
sendSum = [
    *map(sum, zip(sendN.ravel(), sendS.ravel(), sendE.ravel(), sendW.ravel()))
]
log(sendSum[:100])
log(f"{np.mean(sendSum)=} {np.std(sendSum)=} {sps.sem(sendSum)=}")
log(f"{np.median(sendSum)=} {np.min(sendSum)=} {np.max(sendSum)=}")

log("tscControl values ==========================================")
memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
out_tensors = np.zeros((nCol, nRow, tscSizeWords // 2), np.uint32)

runner.memcpy_d2h(
    out_tensors.ravel(),
    runner.get_id("tscControlBuffer"),
    0,  # x0
    0,  # y0
    nCol,  # width
    nRow,  # height
    tscSizeWords // 2,  # num values
    streaming=False,
    data_type=memcpy_dtype,
    order=MemcpyOrder.ROW_MAJOR,
    nonblock=False,
)
data = out_tensors
tscControl_bytes = [
    inner.view(np.uint8).tobytes() for outer in data for inner in outer
]
tscControl_ints = [
    int.from_bytes(genome, byteorder="little") for genome in tscControl_bytes
]
log(tscControl_ints[:100])

log("tscStart values ============================================")
memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
out_tensors = np.zeros((nCol, nRow, tscSizeWords // 2), np.uint32)

runner.memcpy_d2h(
    out_tensors.ravel(),
    runner.get_id("tscStartBuffer"),
    0,  # x0
    0,  # y0
    nCol,  # width
    nRow,  # height
    tscSizeWords // 2,  # num values
    streaming=False,
    data_type=memcpy_dtype,
    order=MemcpyOrder.ROW_MAJOR,
    nonblock=False,
)
data = out_tensors
tscStart_bytes = [
    inner.view(np.uint8).tobytes() for outer in data for inner in outer
]
tscStart_ints = [
    int.from_bytes(genome, byteorder="little") for genome in tscStart_bytes
]
log(tscStart_ints[:100])

log("tscEnd values ==============================================")
memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
out_tensors = np.zeros((nCol, nRow, tscSizeWords // 2), np.uint32)

runner.memcpy_d2h(
    out_tensors.ravel(),
    runner.get_id("tscEndBuffer"),
    0,  # x0
    0,  # y0
    nCol,  # width
    nRow,  # height
    tscSizeWords // 2,  # num values
    streaming=False,
    data_type=memcpy_dtype,
    order=MemcpyOrder.ROW_MAJOR,
    nonblock=False,
)
data = out_tensors
tscEnd_bytes = [
    inner.view(np.uint8).tobytes() for outer in data for inner in outer
]
tscEnd_ints = [
    int.from_bytes(genome, byteorder="little") for genome in tscEnd_bytes
]
log(tscEnd_ints[:100])

log("tsc diffs ==================================================")
log("--------------------------------------------------------- ticks")
tsc_ticks = [end - start for start, end in zip(tscStart_ints, tscEnd_ints)]
log(tsc_ticks[:100])
log(f"{np.mean(tsc_ticks)=} {np.std(tsc_ticks)=} {sps.sem(tsc_ticks)=}")

log("------------------------------------------------------ seconds")
tsc_sec = [diff / tscTicksPerSecond for diff in tsc_ticks]
log(tsc_sec[:100])
log(f"{np.mean(tsc_sec)=} {np.std(tsc_sec)=} {sps.sem(tsc_sec)=}")

log("-------------------------------------------- seconds per cycle")
tsc_cysec = [sec / ncy for (sec, ncy) in zip(tsc_sec, cycle_counts)]
log(tsc_cysec[:100])
log(f"{np.mean(tsc_cysec)=} {np.std(tsc_cysec)=} {sps.sem(tsc_cysec)=}")

log("-------------------------------------------------- cycle hertz")
tsc_cyhz = [1 / cysec for cysec in tsc_cysec]
log(tsc_cyhz[:100])
log(f"{np.mean(tsc_cyhz)=} {np.std(tsc_cyhz)=} {sps.sem(tsc_cyhz)=}")

log("------------------------------------------------- ns per cycle")
tsc_cyns = [cysec * 1e9 for cysec in tsc_cysec]
log(tsc_cyns[:100])
log(f"{np.mean(tsc_cyns)=} {np.std(tsc_cyns)=} {sps.sem(tsc_cyns)=}")

log("perf ======================================================")
# save performance metrics to a file
df = pl.DataFrame({
    "tsc ticks": pl.Series(tsc_ticks, dtype=pl.UInt64),
    "tsc seconds": pl.Series(tsc_sec, dtype=pl.Float32),
    "tsc seconds per cycle": pl.Series(tsc_cysec, dtype=pl.Float32),
    "tsc cycle hertz": pl.Series(tsc_cyhz, dtype=pl.Float32),
    "tsc ns per cycle": pl.Series(tsc_cyns, dtype=pl.Float32),
    "recv sum": pl.Series(recvSum, dtype=pl.UInt32),
    "send sum": pl.Series(sendSum, dtype=pl.UInt32),
    "cycle count": pl.Series(cycle_counts, dtype=pl.UInt32),
    "tsc start": pl.Series(tscStart_ints, dtype=pl.UInt64),
    "tsc end": pl.Series(tscEnd_ints, dtype=pl.UInt64),
    "send N": pl.Series(sendN.ravel(), dtype=pl.UInt32),
    "send S": pl.Series(sendS.ravel(), dtype=pl.UInt32),
    "send E": pl.Series(sendE.ravel(), dtype=pl.UInt32),
    "send W": pl.Series(sendW.ravel(), dtype=pl.UInt32),
    "recv N": pl.Series(recvN.ravel(), dtype=pl.UInt32),
    "recv S": pl.Series(recvS.ravel(), dtype=pl.UInt32),
    "recv E": pl.Series(recvE.ravel(), dtype=pl.UInt32),
    "recv W": pl.Series(recvW.ravel(), dtype=pl.UInt32),
    "tile": pl.Series(whoami_data.ravel(), dtype=pl.UInt32),
    "row": pl.Series(whereami_y_data.ravel(), dtype=pl.UInt16),
    "col": pl.Series(whereami_x_data.ravel(), dtype=pl.UInt16),
})
df.with_columns([
    pl.lit(value, dtype=dtype).alias(key)
    for key, (value, dtype) in metadata.items()
])
write_parquet_verbose(
    df,
    "a=perf"
    f"+flavor={genomeFlavor}"
    f"+seed={globalSeed}"
    f"+ncycle={nCycleAtLeast}"
    "+ext=.pqt",
)
del df, tsc_ticks, tsc_sec, tsc_cysec, tsc_cyhz, tsc_cyns, tscStart_ints, tscEnd_ints

# runner.dump("corefile.cs1")
runner.stop()

# Ensure that the result matches our expectation
log("SUCCESS!")
