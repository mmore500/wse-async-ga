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
echo "find and process job IDs -----------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
# Find all wsjob_id.txt files from previous run step
WORKDIR_RUN="${WORKDIR}/02-run"
echo "WORKDIR_RUN ${WORKDIR_RUN}"

if [ ! -d "${WORKDIR_RUN}" ]; then
    echo "ERROR: Run workdir not found: ${WORKDIR_RUN}"
    echo "Please run 02-run.sh first"
    exit 1
fi

for WSJOB_ID_FILE in "${WORKDIR_RUN}"/*/out/wsjob_id.txt; do
    if [ ! -f "${WSJOB_ID_FILE}" ]; then
        echo "No wsjob_id.txt files found in ${WORKDIR_RUN}/*/out/"
        continue
    fi

    CONFIG_DIR="$(dirname "$(dirname "${WSJOB_ID_FILE}")")"
    CONFIG_NAME="$(basename "${CONFIG_DIR}")"

    echo
    echo "====== Processing config: ${CONFIG_NAME} ======"

    CONFIG_RESULTDIR="${RESULTDIR_STEP}/${CONFIG_NAME}"
    mkdir -p "${CONFIG_RESULTDIR}"

    WSJOB_ID="$(cat "${WSJOB_ID_FILE}")"
    echo "WSJOB_ID ${WSJOB_ID}"

    if [ -n "${WSJOB_ID}" ]; then
        echo "running csctl get job ${WSJOB_ID}..."
        PRE_TIMESTAMP="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        CSCTL_YAML_OUT="$(csctl get job "${WSJOB_ID}" -oyaml 2>&1 || :)"
        CSCTL_TXT_OUT="$(csctl get job "${WSJOB_ID}" 2>&1 || :)"
        POST_TIMESTAMP="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

        # Write yaml with timestamps
        {
            echo "pre_timestamp: \"${PRE_TIMESTAMP}\""
            echo "post_timestamp: \"${POST_TIMESTAMP}\""
            echo "${CSCTL_YAML_OUT}"
        } > "${CONFIG_RESULTDIR}/csctl-job.yaml"

        # Write txt with timestamps as first and last lines
        {
            echo "timestamp: ${PRE_TIMESTAMP}"
            echo "${CSCTL_TXT_OUT}"
            echo "timestamp: ${POST_TIMESTAMP}"
        } > "${CONFIG_RESULTDIR}/csctl-job.txt"

        echo "... done!"
        echo "Output written to:"
        echo "  - ${CONFIG_RESULTDIR}/csctl-job.yaml"
        echo "  - ${CONFIG_RESULTDIR}/csctl-job.txt"
    else
        echo "WSJOB_ID is empty, skipping csctl"
    fi
done

###############################################################################
echo
echo "closeout ---------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
find "${RESULTDIR_STEP}" | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/"
du -ah "${RESULTDIR_STEP}"/* 2>/dev/null || echo "(no files)"

env > "${RESULTDIR_STEP}/env.txt"

###############################################################################
echo
echo "done! ------------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################

echo ">>>fin<<<"
