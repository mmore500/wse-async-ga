#!/usr/bin/env bash

set -euo pipefail

LOG_ALL="$(mktemp)"
LOG_ERR="$(mktemp)"
LOG_OUT="$(mktemp)"

# adapted from https://unix.stackexchange.com/a/61936/605206
exec > >(tee >(tee "${LOG_ALL}" >>"${LOG_OUT}")) \
     2> >(tee >(tee -a "${LOG_ALL}" >>"${LOG_ERR}") >&2)
on_exit() {
    echo
    echo "exit trap ----------------------------------------------------------"
    echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"

    LOGDIR="${HOME}/log/wse-async-ga/"
    echo "copying script and logs to LOGDIR ${LOGDIR}"
    mkdir -p "${LOGDIR}"

    cp "${FLOWDIR}/$(basename "$0")" \
        "${LOGDIR}/flow=${FLOWNAME}+step=${STEPNAME}+what=script+ext=.sh" || :
    cp "${LOG_ALL}" \
        "${LOGDIR}/flow=${FLOWNAME}+step=${STEPNAME}+what=stdall+ext=.log" || :
    cp "${LOG_ERR}" \
        "${LOGDIR}/flow=${FLOWNAME}+step=${STEPNAME}+what=stderr+ext=.log" || :
    cp "${LOG_OUT}" \
        "${LOGDIR}/flow=${FLOWNAME}+step=${STEPNAME}+what=stdout+ext=.log" || :

    echo "copying script and logs to RESULTDIR_STEP ${RESULTDIR_STEP}"
    cp "${FLOWDIR}/$(basename "$0")" "${RESULTDIR_STEP}/" || :
    cp "${LOG_ALL}" "${RESULTDIR_STEP}/stdall.log" || :
    cp "${LOG_ERR}" "${RESULTDIR_STEP}/stderr.log" || :
    cp "${LOG_OUT}" "${RESULTDIR_STEP}/stdout.log" || :
}
trap on_exit EXIT

FLOWDIR="$(realpath "$(dirname "$0")")"
FLOWNAME="$(basename "${FLOWDIR}")"
STEPNAME="$(basename "$0" .sh)"
WORKDIR="${FLOWDIR}/workdir"
RESULTDIR="${FLOWDIR}/resultdir"

###############################################################################
echo
echo
echo "============================================= ${FLOWNAME} :: ${STEPNAME}"
###############################################################################
source "${HOME}/.env" || true

###############################################################################
echo
echo "log context ------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
echo "date $(date '+%Y-%m-%d %H:%M:%S')"
echo "hostname $(hostname)"
echo "SECONDS ${SECONDS}"

echo "STEPNAME ${STEPNAME}"
echo "FLOWDIR ${FLOWDIR}"
echo "FLOWNAME ${FLOWNAME}"
echo "WORKDIR ${WORKDIR}"
echo "RESULTDIR ${RESULTDIR}"

echo "LOG_ERR ${LOG_ERR}"
echo "LOG_OUT ${LOG_OUT}"
echo "LOG_ALL ${LOG_ALL}"

###############################################################################
echo
echo "make step work dir -----------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
WORKDIR_STEP="${WORKDIR}/${STEPNAME}"
echo "WORKDIR_STEP ${WORKDIR_STEP}"
if [ -d "${WORKDIR_STEP}" ]; then
    echo "Clearing WORKDIR_STEP ${WORKDIR_STEP}"
    rm -rf "${WORKDIR_STEP}"
fi
mkdir -p "${WORKDIR_STEP}"

###############################################################################
echo
echo "make step result dir ---------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
RESULTDIR_STEP="${RESULTDIR}/${STEPNAME}"
echo "RESULTDIR_STEP ${RESULTDIR_STEP}"
if [ -d "${RESULTDIR_STEP}" ]; then
    echo "Clearing RESULTDIR_STEP ${RESULTDIR_STEP}"
    rm -rf "${RESULTDIR_STEP}"
fi
mkdir -p "${RESULTDIR_STEP}"

###############################################################################
echo
echo "setup venv  ------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
VENVDIR="${WORKDIR}/venv3x"
echo "VENVDIR ${VENVDIR}"

echo "creating venv"
source "${VENVDIR}/bin/activate"
python3 -m uv pip freeze | tee "${RESULTDIR_STEP}/pip-freeze.txt"

###############################################################################
echo
echo "log source -------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
echo "log revision"
git -C "${FLOWDIR}" rev-parse HEAD > "${RESULTDIR_STEP}/git-revision.txt"
echo "log remote"
git -C "${FLOWDIR}" remote -v > "${RESULTDIR_STEP}/git-remote.txt"
echo "log status"
git -C "$(git -C "${FLOWDIR}" rev-parse --show-toplevel)" status \
    > "${RESULTDIR_STEP}/git-status.txt"
echo "log diff"
git -C "${FLOWDIR}" --no-pager diff > "${RESULTDIR_STEP}/git-status.diff" || :
git -C "${FLOWDIR}" ls-files -z --others --exclude-standard | xargs -0 -I {} git -C "${FLOWDIR}" --no-pager diff --no-index /dev/null {} >> "${RESULTDIR_STEP}/git-status.diff" || :

SRCDIR="${WORKDIR}/src"
echo "SRCDIR ${SRCDIR}"
echo "log revision"
git -C "${SRCDIR}" rev-parse HEAD > "${RESULTDIR_STEP}/src-revision.txt"
echo "log remote"
git -C "${SRCDIR}" remote -v > "${RESULTDIR_STEP}/src-remote.txt"
echo "log status"
git -C "$(git -C "${SRCDIR}" rev-parse --show-toplevel)" status \
    > "${RESULTDIR_STEP}/src-status.txt"
echo "log diff"
git -C "${SRCDIR}" --no-pager diff > "${RESULTDIR_STEP}/src-status.diff" || :
git -C "${SRCDIR}" ls-files -z --others --exclude-standard | xargs -0 -I {} git -C "${SRCDIR}" --no-pager diff --no-index /dev/null {} >> "${RESULTDIR_STEP}/src-status.diff" || :

###############################################################################
echo
echo "minify phylogeny -------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
PHYLOFRAME_CONTAINER="docker://ghcr.io/mmore500/phyloframe:v0.12.1"
echo "PHYLOFRAME_CONTAINER ${PHYLOFRAME_CONTAINER}"
# Minify each run's phylogeny independently, writing outputs flat into the
# step directory, distinguished by seed.
for CONFIG_DIR in "${WORKDIR}/04-build-phylo"/*/; do
    CONFIG_NAME="$(basename "${CONFIG_DIR}")"
    echo
    echo "====== Minifying phylogeny for config: ${CONFIG_NAME} ======"
    echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"

    PHYLO_SOURCE="${CONFIG_DIR}a=phylogeny+ext=.pqt"
    echo "PHYLO_SOURCE ${PHYLO_SOURCE}"
    SEED="$(echo "${CONFIG_NAME}" | grep -oP '(?<=seed=)[0-9]+')"
    echo "SEED ${SEED}"
    # outputs are flat, so seed must uniquely identify each config
    if compgen -G "${WORKDIR_STEP}/a=*+seed=${SEED}+*" > /dev/null; then
        echo "duplicate seed ${SEED} for config ${CONFIG_NAME}" >&2
        exit 1
    fi
    PHYLO_MINIFIED="${WORKDIR_STEP}/a=phylogeny+what=minified-nocollapse+seed=${SEED}+ext=.pqt"
    echo "PHYLO_MINIFIED ${PHYLO_MINIFIED}"
    CONFIG_JSON="${WORKDIR_STEP}/a=config+what=minified-nocollapse+seed=${SEED}+ext=.json"
    echo "CONFIG_JSON ${CONFIG_JSON}"

    # document run configuration alongside minified phylogeny, including
    # compile environment (01-compile env.sh exports) and compconf data
    WORKDIR_COMPILE="${WORKDIR}/01-compile/${CONFIG_NAME}"
    python3 - \
        "${CONFIG_NAME}" "${FLOWNAME}" "${SEED}" "$(basename "${PHYLO_MINIFIED}")" \
        "${WORKDIR_COMPILE}/env.sh" "${WORKDIR_COMPILE}/compconf.json" \
        > "${CONFIG_JSON}" << 'PYEOF'
import json
import shlex
import sys

config, flow, seed, phylogeny, env_path, compconf_path = sys.argv[1:]
env = dict(
    token.split("=", 1)
    for line in open(env_path)
    if line.startswith("export ")
    for token in shlex.split(line)[1:]
)
json.dump(
    {
        "config": config,
        "flow": flow,
        "seed": int(seed),
        "phylogeny": phylogeny,
        "env": env,
        "compconf": json.load(open(compconf_path)),
    },
    sys.stdout,
    indent=2,
)
print()
PYEOF
    cat "${CONFIG_JSON}"

    # Space-saving re-encoding (see phylogeny-space-optimizations.md); keeps
    # singleton roots. Lossy: drops data_hex and dstream_data_id. Unlike
    # 05-minify-phylo, does NOT collapse monomorphic clades; every genome is kept.
    # 1. pack 16 trait_num* (0/1/2) columns into two UInt64 bitmasks; rebuild as
    #    trait_num{i} = bit i of trait_present * (bit i of trait_focal + 1)
    #    (null trait_present marks taxa without a genome)
    # 2. keep only selected columns, dropping ones redundant with others or
    #    rebuildable on load (trait_byte{B}_bit{b}: trait_num{8B+b};
    #    bookend_right/value: bookend_left; hstrat_rank: origin_time;
    #    position: row * nCol + col; (non)focal_trait_count: popcount of
    #    trait_present & (~)trait_focal), plus data_hex, dstream_data_id,
    #    bookend_left, and flag masks
    # 3. add clade_size (1 for genomes, else 0) and last_origin_time (own
    #    origin_time), for column compatibility with collapsed output
    # 4. fill nulls in single-valued (e.g., config) columns
    # 5. reorder rows depth-first, replacing id/ancestor_id with ancestor_offset;
    #    rebuild as id = row index, ancestor_id = id - ancestor_offset
    # 6. write with zstd level 19
    echo "${PHYLO_SOURCE}" \
        | singularity exec "${PHYLOFRAME_CONTAINER}" \
            python3 -m phyloframe.legacy._alifestd_pipe_unary_ops_polars \
            "${PHYLO_MINIFIED}" \
            --with-column 'pl.col("data_hex").is_not_null().alias("is_genome")' \
            --with-column 'pl.when(pl.col("data_hex").is_not_null()).then(
                pl.sum_horizontal(
                    (pl.col(f"trait_num{i}") != 0).cast(pl.UInt64)
                    * pl.lit(1 << i, dtype=pl.UInt64)
                    for i in range(16)
                )
            ).alias("trait_present")' \
            --with-column 'pl.when(pl.col("data_hex").is_not_null()).then(
                pl.sum_horizontal(
                    pl.col(f"flag_is_focal_mask_byte{B}").cast(pl.UInt64)
                    * pl.lit(1 << (8 * B), dtype=pl.UInt64)
                    for B in range(2)
                )
            ).alias("trait_focal")' \
            --select "id" \
            --select "ancestor_id" \
            --select "origin_time" \
            --select "dstream_rank" \
            --select "dstream_S" \
            --select "hstrat_differentia_bitwidth" \
            --select "is_extant" \
            --select "is_genome" \
            --select "trait_present" \
            --select "trait_focal" \
            --select "layer" \
            --select "layer_T" \
            --select "row" \
            --select "col" \
            --select "subgrid" \
            --select "subgrid_row" \
            --select "subgrid_col" \
            --select "config" \
            --select "file" \
            --select "replicate" \
            --select "sample_per_stratum" \
            --select "genomeFlavor" \
            --select "populationFlavor" \
            --select "memcpyMajor" \
            --select "globalSeed" \
            --select "^n(Col|Row|Wav|Trait|Cycle|ColSubgrid|RowSubgrid)$" \
            --select "nonBlock" \
            --select "tilePopSize" \
            --select "tournSize" \
            --select "msec" \
            --select "tsc" \
            --select "dstream_algo" \
            --select "^dstream_(storage|T)_bit(offset|width)$" \
            --select "dstream_S_right" \
            --select "^flag_sites?_bit(offset|width)$" \
            --select "^(mls|deleterious|beneficial)_(transmission_reduction|withinhost_gain|withinhost_loss)_p_byte\d+_bit\d+$" \
            --select "mls_host_clear_p" \
            --select "mls_withinhost_neutral_p" \
            --select "_compconf_dummy" \
            --select "^CEREBRASLIB_.*$" \
            --select "^ASYNC_GA_.*$" \
            --op 'lambda df: df.collect()' \
            --op 'lambda df: df.with_columns(
                clade_size=pl.col("is_genome").cast(pl.UInt32),
                last_origin_time=pl.col("origin_time"),
            )' \
            --op 'lambda df: df.with_columns(
                pl.col(col).fill_null(df[col].drop_nulls().first())
                for col in df.columns
                if df[col].has_nulls() and df[col].drop_nulls().n_unique() == 1
            )' \
            --op 'pfl.alifestd_assign_contiguous_ids_polars' \
            --op 'pfl.alifestd_minify_polars' \
            --shrink-dtypes \
            --eager-read --eager-write \
            --write-kwarg 'compression="zstd"' \
            --write-kwarg 'compression_level=19' \
            | tee "${RESULTDIR_STEP}/a=log+what=minified-nocollapse+seed=${SEED}+ext=.log"

    ls -l "${PHYLO_SOURCE}" "${PHYLO_MINIFIED}" \
        | tee "${RESULTDIR_STEP}/a=file-sizes+what=minified-nocollapse+seed=${SEED}+ext=.txt"
done

###############################################################################
echo
echo "closeout ---------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
find "${WORKDIR_STEP}" | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/"
du -ah "${WORKDIR_STEP}"/*

cp -r "${WORKDIR_STEP}"/* "${RESULTDIR_STEP}"

find "${RESULTDIR_STEP}" | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/"
du -ah "${RESULTDIR_STEP}"/*

env > "${RESULTDIR_STEP}/env.txt"

###############################################################################
echo
echo "done! ------------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################

echo ">>>fin<<<"
