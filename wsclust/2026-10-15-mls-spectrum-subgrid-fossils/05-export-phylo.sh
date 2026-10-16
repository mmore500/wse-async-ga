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
echo "export phylogeny -------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
PHYLOFRAME_CONTAINER="docker://ghcr.io/mmore500/phyloframe:v0.2.1"
JOINEM_CONTAINER="docker://ghcr.io/mmore500/joinem:v0.16.0"
# downsample, collapse unifurcations, assign contiguous ids, export newick
# usage: downsample_and_export <label> <phyloframe module> <module args...>
downsample_and_export() {
    local label="$1" module="$2"
    shift 2
    local outpqt="${CONFIG_WORKDIR}/a=phylogeny+${label}+ext=.pqt"
    echo "downsample ${label} ----------------------------------------------"

    echo "${CONFIG_WORKDIR}/a=phylogeny+ext=.pqt" \
        | singularity exec --env POLARS_ENGINE_AFFINITY="streaming" \
            "${PHYLOFRAME_CONTAINER}" \
            python3 -m "phyloframe.legacy.${module}" \
            "${outpqt}" "$@" --eager-write \
            --with-column 'pl.col("id").alias("original_id")' \
            | tee "${CONFIG_RESULTDIR}/${module}_${label}.log"

    echo "${outpqt}" \
        | singularity exec "${PHYLOFRAME_CONTAINER}" \
            python3 -m phyloframe.legacy._alifestd_collapse_unifurcations_polars \
            "${outpqt}" --eager-write

    echo "${outpqt}" \
        | singularity exec "${PHYLOFRAME_CONTAINER}" \
            python3 -m phyloframe.legacy._alifestd_assign_contiguous_ids_polars \
            "${outpqt}" --eager-write

    singularity exec "${PHYLOFRAME_CONTAINER}" \
        python3 -m phyloframe.legacy._alifestd_as_newick_polars \
            -i "${outpqt}" \
            -o "${CONFIG_WORKDIR}/a=phylotree+${label}+ext=.nwk" \
            | tee "${CONFIG_RESULTDIR}/_alifestd_as_newick_polars_${label}.log"

    echo "${outpqt}" \
        | singularity run "${JOINEM_CONTAINER}" \
            "${CONFIG_WORKDIR}/a=phylometa+${label}+ext=.csv" \
            --select "id" \
            --select "original_id" \
            --select "origin_time" \
            --select "focal_trait_count" \
            --select "nonfocal_trait_count" \
            --select "layer" \
            --select "row" \
            --select "col" \
            --select "subgrid" \
            --select "subgrid_row" \
            --select "subgrid_col" \
            --select "^trait_byte\d+_bit\d+$" \
            --select "^trait_num\d+$" \
            | tee "${CONFIG_RESULTDIR}/joinem_${label}.log"

    gzip -k "${CONFIG_WORKDIR}/a=phylotree+${label}+ext=.nwk"
    gzip -k "${CONFIG_WORKDIR}/a=phylometa+${label}+ext=.csv"
}

# 04-build-phylo writes one file holding an independent tree per run, keyed
# by `config`; split it out and export each run's tree separately
PHYLO_SOURCE="${WORKDIR}/04-build-phylo/a=phylogeny+ext=.pqt"
echo "PHYLO_SOURCE ${PHYLO_SOURCE}"
for CONFIG_DIR in "${WORKDIR}/03-postprocess"/*/; do
    CONFIG_NAME="$(basename "${CONFIG_DIR}")"
    echo
    echo "====== Exporting phylogeny for config: ${CONFIG_NAME} ======"

    CONFIG_WORKDIR="${WORKDIR_STEP}/${CONFIG_NAME}"
    CONFIG_RESULTDIR="${RESULTDIR_STEP}/${CONFIG_NAME}"
    mkdir -p "${CONFIG_WORKDIR}"
    mkdir -p "${CONFIG_RESULTDIR}"

    echo "PHYLOFRAME_CONTAINER ${PHYLOFRAME_CONTAINER}"

    # workaround for pyarrow.lib.ArrowTypeError:
    # Converting unsigned dictionary indices to pandas not yet supported, index type: uint32
    echo "${PHYLO_SOURCE}" \
        | singularity run "${JOINEM_CONTAINER}" \
        --eager-read --eager-write \
        --with-column 'pl.selectors.categorical().cast(pl.String)' \
        --filter "pl.col(\"config\") == \"${CONFIG_NAME}\"" \
        "${CONFIG_WORKDIR}/a=phylogeny+ext=.pqt"

    # tree ids were reassigned across runs; make this run's contiguous
    echo "${CONFIG_WORKDIR}/a=phylogeny+ext=.pqt" \
        | singularity exec "${PHYLOFRAME_CONTAINER}" \
            python3 -m phyloframe.legacy._alifestd_assign_contiguous_ids_polars \
            "${CONFIG_WORKDIR}/a=phylogeny+ext=.pqt" --eager-write

    singularity exec "${PHYLOFRAME_CONTAINER}" \
        python3 -m phyloframe.legacy._alifestd_as_newick_polars \
            -i "${CONFIG_WORKDIR}/a=phylogeny+ext=.pqt" \
            -o "${CONFIG_WORKDIR}/a=phylotree+ext=.nwk" \
            | tee "${CONFIG_RESULTDIR}/_alifestd_as_newick_polars.log"

    echo "${CONFIG_WORKDIR}/a=phylogeny+ext=.pqt" \
        | singularity run "${JOINEM_CONTAINER}" \
            "${CONFIG_WORKDIR}/a=phylometa+ext=.csv" \
            --select "id" \
            --select "origin_time" \
            --select "focal_trait_count" \
            --select "nonfocal_trait_count" \
            --select "layer" \
            --select "row" \
            --select "col" \
            --select "subgrid" \
            --select "subgrid_row" \
            --select "subgrid_col" \
            --select "^trait_byte\d+_bit\d+$" \
            --select "^trait_num\d+$" \
            | tee "${CONFIG_RESULTDIR}/joinem.log"

    gzip -k "${CONFIG_WORKDIR}/a=phylotree+ext=.nwk"
    gzip -k "${CONFIG_WORKDIR}/a=phylometa+ext=.csv"

    ###############################################################################
    echo
    echo "cladesample phylogeny --------------------------------------------------"
    echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
    ###############################################################################
    for csamp in 8192 32768 131072 262144; do
        downsample_and_export "csamp=${csamp}" \
            _alifestd_downsample_tips_clade_polars -n "${csamp}" --seed 1
    done

    ###############################################################################
    echo
    echo "downsample phylogeny ---------------------------------------------------"
    echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
    ###############################################################################
    for dsamp in 8192 32768 131072 262144; do
        downsample_and_export "dsamp=${dsamp}" \
            _alifestd_downsample_tips_polars -n "${dsamp}" --seed 1
    done
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
