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
echo "build phylogeny --------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
export APPTAINERENV_POLARS_MAX_THREADS=32
export APPTAINERENV_NUMBA_NUM_THREADS=32
echo "APPTAINERENV_POLARS_MAX_THREADS=${APPTAINERENV_POLARS_MAX_THREADS}"
echo "APPTAINERENV_NUMBA_NUM_THREADS=${APPTAINERENV_NUMBA_NUM_THREADS}"

export SINGULARITYENV_POLARS_MAX_THREADS="${APPTAINERENV_POLARS_MAX_THREADS}"
export SINGULARITYENV_NUMBA_NUM_THREADS="${APPTAINERENV_NUMBA_NUM_THREADS}"
echo "SINGULARITYENV_POLARS_MAX_THREADS=${SINGULARITYENV_POLARS_MAX_THREADS}"
echo "SINGULARITYENV_NUMBA_NUM_THREADS=${SINGULARITYENV_NUMBA_NUM_THREADS}"

# Reconstruct from fossils only (a=fossils+..., not end-state a=genomes+...).
# Sample non-empty fossil genomes uniformly within each (layer, subgrid)
# stratum, taking an equal share of SAMPLE_BUDGET per stratum (at least one).
# note: --sample-stratify requires hstrat >=v1.25.3 (bundles joinem >=v0.15.0)
HSTRAT_CONTAINER="docker://ghcr.io/mmore500/hstrat:v1.25.3"
echo "HSTRAT_CONTAINER ${HSTRAT_CONTAINER}"
SAMPLE_BUDGET=4000000
echo "SAMPLE_BUDGET ${SAMPLE_BUDGET}"
NONEMPTY='~pl.col("data_hex").str.contains(r"^0+$")'
echo "NONEMPTY ${NONEMPTY}"

for CONFIG_DIR in "${WORKDIR}/03-postprocess"/*/; do
    CONFIG_NAME="$(basename "${CONFIG_DIR}")"
    echo
    echo "====== Building phylogeny for config: ${CONFIG_NAME} ======"

    CONFIG_WORKDIR="${WORKDIR_STEP}/${CONFIG_NAME}"
    CONFIG_RESULTDIR="${RESULTDIR_STEP}/${CONFIG_NAME}"
    mkdir -p "${CONFIG_WORKDIR}"
    mkdir -p "${CONFIG_RESULTDIR}"

    FOSSILS="$(find "${CONFIG_DIR}" -maxdepth 1 -type f -name 'a=fossils+*.pqt')"
    echo "FOSSILS ${FOSSILS}"
    NUM_STRATA="$(
        python3 -c 'import polars as pl, sys; print(
            pl.scan_parquet(sys.argv[1]).filter(eval(sys.argv[2]))
            .select(pl.struct("layer", "subgrid").n_unique()).collect().item()
        )' "${FOSSILS}" "${NONEMPTY}"
    )"
    PER_STRATUM=$(( SAMPLE_BUDGET / NUM_STRATA > 0 ? SAMPLE_BUDGET / NUM_STRATA : 1 ))
    echo "NUM_STRATA ${NUM_STRATA} PER_STRATUM ${PER_STRATUM}"

    echo "${FOSSILS}" \
        | singularity exec "${HSTRAT_CONTAINER}" \
        python3 -O -m hstrat.dataframe.surface_build_tree \
            "${CONFIG_WORKDIR}/a=phylogeny+ext=.pqt" \
            --trie-postprocessor \
                'hstrat.AssignOriginTimeNodeRankTriePostprocessor(t0="dstream_S")' \
            --shuffle-over-same-T-seed 1 \
            --mp-pool-size 8 \
            --no-drop-dstream-metadata \
            --collapse-unif-freq 50 \
            --exploded-slice-size 100000 \
            --shrink-dtypes \
            --write-kwarg 'compression="lz4"' \
            --filter '~pl.col("is_extant")' \
            --filter "${NONEMPTY}" \
            --sample "pl.len().clip(upper_bound=${PER_STRATUM})" \
            --sample-stratify 'pl.col("layer")' \
            --sample-stratify 'pl.col("subgrid")' \
            --seed 1 \
            --how "diagonal_relaxed" \
            --eager-read --eager-write --string-cache \
            --with-column 'pl.lit(filepath).cast(pl.Categorical).alias("file")' \
            --with-column 'pl.sum_horizontal(
                ((pl.col("data_hex").str.slice(2*b, 2).str.to_integer(base=16)
                ^ pl.col(f"flag_nand_mask_byte{b}"))
                & pl.col(f"flag_is_focal_mask_byte{b}")).bitwise_count_ones()
                for b in range(8)
            ).alias("focal_trait_count")' \
            --with-column 'pl.sum_horizontal(
                ((pl.col("data_hex").str.slice(2*b, 2).str.to_integer(base=16)
                ^ pl.col(f"flag_nand_mask_byte{b}"))
                & (~pl.col(f"flag_is_focal_mask_byte{b}"))).bitwise_count_ones()
                for b in range(8)
            ).cast(pl.UInt8).alias("nonfocal_trait_count")' \
            --with-column '*(
                (

                    (((
                        pl.col("data_hex")
                        .str.slice(B * 2, 2)
                        .str.to_integer(base=16)
                        ^ pl.col(f"flag_nand_mask_byte{B}")
                    ) & pl.lit(1 << b)) // pl.lit(1 << b))
                    * pl.when(
                        pl.col(f"flag_is_focal_mask_byte{B}")
                        & pl.lit(1 << b)
                        != 0
                    ).then(2).otherwise(1)
                ).cast(pl.UInt8).alias(alias)
                for B in range(8)
                for b in range(8)
                for alias in (f"trait_byte{B}_bit{b}", f"trait_num{B * 8 + b}")
            )' \
            | tee "${CONFIG_RESULTDIR}/surface_build_tree.log"
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
