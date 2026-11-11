"""Compute train/test loss and phenotype distribution for sampled genomes
from a genome_evodd `a=genomes+...pqt` run output.

genome_evodd's development function produces a single deterministic
phenotype `Pa` per genome (embryo + dev matrix, no external input); fitness
during evolution only ever scores `Pa` against whichever of 3 training
patterns is currently active in the schedule
(S1=AAAA, S2=ABBA, S3=BBAA -- see cerebraslib/genome/genome_evodd.csl and
https://github.com/mmore500/evodd/blob/main/bindle/2026-07-23-exploratory.py).
Those 3 patterns are 3 of the 2**4 = 16 possible combinations of 4
independent length-4 sub-modules (each in state "A" or "B"); this script
develops each sampled genome once and scores that single phenotype against
*all* 16 possible combinations, split into:
  - "training" loss/hits: the 3 patterns actually used for selection
  - "testing" loss/hits: the remaining 13 patterns, held out from selection,
    used here purely as a post-hoc generalization/canalization probe

Phenotype classification is via exact sign-match: a genome's phenotype is
classified as pattern i if sign(Pa) == sign(pattern_i) elementwise; if it
doesn't exactly match any of the 16, it is classified "other". Each
genome's classification is reported as one-hot indicator columns (one
column per of the 16 patterns, plus an "other" column) directly on the
per-genome output table -- one row per sampled genome -- rather than as a
separate pattern-indexed distribution table. The population-level fraction
in each category is recoverable by averaging the corresponding column
(e.g. `df["AAAA"].mean()`); "unique other" phenotypes are still tallied
and printed as a summary statistic, since that's not recoverable from the
one-hot columns alone.
Loss is mean squared error between Pa and each candidate pattern.
"""

import argparse
import itertools
import sys
import time

import numpy as np
import polars as pl
from tqdm import tqdm

# genome_evodd GRN development constants (must match
# cerebraslib/genome/genome_evodd.csl defaults) ==============================
N = 16
T_DEVELOP = 10
TAU1 = np.float16(1.0)
TAU2 = np.float16(0.2)
ALPHA = np.float16(0.5)


def make_pattern(states: str) -> np.ndarray:
    mod_a = [
        np.array([-1, 1, -1, 1], dtype=np.float16),
        np.array([-1, -1, 1, 1], dtype=np.float16),
        np.array([-1, 1, 1, -1], dtype=np.float16),
        np.array([-1, -1, -1, -1], dtype=np.float16),
    ]
    mod_b = [-m for m in mod_a]
    mods = [mod_a[i] if s == "A" else mod_b[i] for i, s in enumerate(states)]
    return np.concatenate(mods)


def build_pattern_table() -> tuple[np.ndarray, list[str], list[int]]:
    labels = ["".join(c) for c in itertools.product("AB", repeat=4)]
    patterns = np.stack([make_pattern(label) for label in labels])
    training_labels = ["AAAA", "ABBA", "BBAA"]  # S1, S2, S3
    training_indices = [labels.index(t) for t in training_labels]
    return patterns, labels, training_indices


def develop(embryo: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    p = embryo.astype(np.float16).copy()
    for _ in range(T_DEVELOP):
        inter = (matrix.astype(np.float16) @ p).astype(np.float16)
        p = (
            p
            + TAU1 * np.tanh((ALPHA * inter).astype(np.float16)).astype(np.float16)
            - TAU2 * p
        ).astype(np.float16)
    return p


def decode_genome(
    data_hex: str, embryo_off: int, embryo_n: int, matrix_off: int, matrix_n: int
) -> tuple[np.ndarray, np.ndarray]:
    # inverts client.py's `astype(">u4")` + byte-view hex serialization
    words = np.frombuffer(bytes.fromhex(data_hex), dtype=">u4")
    lo = (words & 0xFFFF).astype(np.uint16)
    hi = ((words >> 16) & 0xFFFF).astype(np.uint16)
    f16pairs = np.empty(len(words) * 2, dtype=np.uint16)
    f16pairs[0::2] = lo
    f16pairs[1::2] = hi
    f16 = f16pairs.view(np.float16)
    embryo = f16[embryo_off : embryo_off + embryo_n]
    matrix = f16[matrix_off : matrix_off + matrix_n].reshape(16, 16)
    return embryo, matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="path to a=genomes+...pqt run output"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10_000,
        help="number of genomes to randomly sample and analyze (default: 10000)",
    )
    parser.add_argument("--seed", type=int, default=1, help="sampling RNG seed")
    parser.add_argument(
        "--project-to",
        type=int,
        default=900_000,
        help="population size to project measured throughput/ETA onto",
    )
    parser.add_argument(
        "--per-genome-out",
        default="a=evoddPhenotypes+ext=.pqt",
        help="output path for per-sampled-genome results",
    )
    args = parser.parse_args()

    print(f"loading {args.input}...", flush=True)
    t_load0 = time.time()
    df = pl.read_parquet(args.input)
    t_load1 = time.time()
    print(f"... loaded {len(df)} rows in {t_load1 - t_load0:.2f}s", flush=True)

    n = min(args.sample_size, len(df))
    if n < len(df):
        df = df.sample(n=n, seed=args.seed, shuffle=True)
    print(f"analyzing {n} sampled genomes...", flush=True)

    embryo_off = df["embryo_bitoffset"][0] // 16
    embryo_n = df["embryo_bitwidth"][0] // 16
    matrix_off = df["dev_matrix_bitoffset"][0] // 16
    matrix_n = df["dev_matrix_bitwidth"][0] // 16

    patterns, labels, training_indices = build_pattern_table()
    training_mask = np.zeros(16, dtype=bool)
    training_mask[training_indices] = True
    patterns_f32 = patterns.astype(np.float32)

    train_losses = np.empty(n, dtype=np.float32)
    test_losses = np.empty(n, dtype=np.float32)
    # one-hot columns: one per pattern (16) plus "other", 1 row per genome
    one_hot = np.zeros((n, 17), dtype=np.float32)  # columns 0..15 == labels, 16 == other
    other_signatures: dict[int, int] = {}

    t_compute0 = time.time()
    for i, data_hex in enumerate(
        tqdm(df["data_hex"].to_list(), desc="developing genomes", unit="genome")
    ):
        embryo, matrix = decode_genome(
            data_hex, embryo_off, embryo_n, matrix_off, matrix_n
        )
        phenotype = develop(embryo, matrix).astype(np.float32)

        per_pattern_loss = ((phenotype[None, :] - patterns_f32) ** 2).mean(axis=1)
        train_losses[i] = per_pattern_loss[training_mask].mean()
        test_losses[i] = per_pattern_loss[~training_mask].mean()

        binarized = np.where(phenotype >= 0, 1, -1).astype(np.int8)
        exact_matches = np.flatnonzero(np.all(binarized[None, :] == patterns, axis=1))
        if exact_matches.size:
            one_hot[i, exact_matches[0]] = 1.0
        else:
            one_hot[i, 16] = 1.0
            signature = int(
                "".join("1" if b > 0 else "0" for b in binarized), base=2
            )
            other_signatures[signature] = other_signatures.get(signature, 0) + 1
    t_compute1 = time.time()

    compute_elapsed = t_compute1 - t_compute0
    rate = n / compute_elapsed if compute_elapsed > 0 else float("inf")
    print(
        f"... done in {compute_elapsed:.2f}s "
        f"({rate:.1f} genomes/sec, {1e6 / rate:.2f}us/genome)",
        flush=True,
    )
    projected_seconds = args.project_to / rate
    print(
        f"projected compute time for full population of {args.project_to}: "
        f"{projected_seconds:.1f}s ({projected_seconds / 60:.2f} min) "
        "-- excludes one-time parquet load/write I/O",
        flush=True,
    )

    # per-genome results --------------------------------------------------------
    # one row per sampled genome; one column per phenotype category (16
    # training/testing patterns + "other"), value 1.0 if that genome's
    # developed phenotype exactly sign-matches that category, else 0.0.
    # Column-averaging recovers the population-level fraction in each
    # category, e.g. `per_genome_df["AAAA"].mean()`.
    category_columns = {
        label: one_hot[:, idx] for idx, label in enumerate(labels)
    }
    category_columns["other"] = one_hot[:, 16]
    per_genome_df = pl.DataFrame(
        {
            "train_loss": train_losses,
            "test_loss": test_losses,
            **category_columns,
        }
    )
    per_genome_df.write_parquet(args.per_genome_out, compression="lz4")
    print(f"wrote {args.per_genome_out}", flush=True)

    print("\n=== per-genome results preview ===", flush=True)
    with pl.Config(tbl_cols=20):
        print(per_genome_df.head(10), flush=True)

    n_other = int(one_hot[:, 16].sum())
    print("\n=== category fractions (mean of each one-hot column) ===", flush=True)
    for idx, label in enumerate(labels):
        tag = "training" if training_mask[idx] else "testing"
        print(f"  {label} ({tag}): {one_hot[:, idx].mean():.2%}", flush=True)
    print(f"  other: {one_hot[:, 16].mean():.2%}", flush=True)

    print("\n=== other (non-canalized) phenotypes ===", flush=True)
    print(f"fraction other: {n_other / n:.2%} ({n_other}/{n})", flush=True)
    print(f"unique other phenotype signatures: {len(other_signatures)}", flush=True)

    print("\n=== loss summary ===", flush=True)
    print(
        f"train loss (mean over 3 training patterns): "
        f"mean={train_losses.mean():.4f} median={np.median(train_losses):.4f} "
        f"std={train_losses.std():.4f}",
        flush=True,
    )
    print(
        f"test loss (mean over 13 held-out patterns): "
        f"mean={test_losses.mean():.4f} median={np.median(test_losses):.4f} "
        f"std={test_losses.std():.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
