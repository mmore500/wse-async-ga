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
SRCDIR="${WORKDIR}/src"


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
echo "SRCDIR ${SRCDIR}"

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
echo "setup venv -------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
VENVDIR="${WORKDIR}/venv3x"
echo "VENVDIR ${VENVDIR}"

echo "creating venv"
rm -rf "${VENVDIR}"
python3 -m venv "${VENVDIR}"
source "${VENVDIR}/bin/activate"
which python3

echo "setting up venv"
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade uv
python3 -m uv pip install -r "${SRCDIR}/requirements.txt"
python3 -m uv pip freeze | tee "${RESULTDIR_STEP}/pip-freeze.txt"

###############################################################################
echo
echo "postprocess configs ----------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
WORKDIR_RUN="${WORKDIR}/02-run"
echo "WORKDIR_RUN ${WORKDIR_RUN}"

for CONFIG_DIR in "${WORKDIR_RUN}"/*/; do
    CONFIG_NAME="$(basename "${CONFIG_DIR}")"
    echo
    echo "====== Postprocessing config: ${CONFIG_NAME} ======"

    CONFIG_WORKDIR="${WORKDIR_STEP}/${CONFIG_NAME}"
    CONFIG_RESULTDIR="${RESULTDIR_STEP}/${CONFIG_NAME}"
    mkdir -p "${CONFIG_WORKDIR}"
    mkdir -p "${CONFIG_RESULTDIR}"

    ###########################################################################
    echo
    echo "setup symlinks: ${CONFIG_NAME} -----------------------------------------"
    echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
    ###########################################################################
    # Symlink raw/ from run step
    ln -sf "${WORKDIR_RUN}/${CONFIG_NAME}/out/raw" "${CONFIG_WORKDIR}/raw"
    ls -la "${CONFIG_WORKDIR}/raw"

    # Symlink other needed files from compile step
    WORKDIR_COMPILE="${WORKDIR}/01-compile/${CONFIG_NAME}"
    ln -sf "${WORKDIR_COMPILE}/out" "${CONFIG_WORKDIR}/out"
    ln -sf "${WORKDIR_COMPILE}/compconf.json" "${CONFIG_WORKDIR}/compconf.json"
    ln -sf "${SRCDIR}/kernel-async-ga/cerebraslib" "${CONFIG_WORKDIR}/cerebraslib"

    ###########################################################################
    echo
    echo "run postprocess: ${CONFIG_NAME} ----------------------------------------"
    echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
    ###########################################################################
    cd "${CONFIG_WORKDIR}"
    echo "PWD ${PWD}"

    # Source env to get ASYNC_GA_* variables
    source "${WORKDIR_COMPILE}/env.sh"

    python3 "${SRCDIR}/kernel-async-ga/client.py" --process-fossils 2>&1 \
        | tee "${CONFIG_WORKDIR}/postprocess.log"

    ###########################################################################
    echo
    echo "closeout: ${CONFIG_NAME} -----------------------------------------------"
    echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
    ###########################################################################
    # Copy output files to result dir
    find "${CONFIG_WORKDIR}" -maxdepth 1 -type f \
        \( -name "*.log" -o -name "*.pqt" -o -name "*.npy" \) \
        -exec cp {} "${CONFIG_RESULTDIR}/" \;

    find "${CONFIG_RESULTDIR}" | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/"
    du -ah "${CONFIG_RESULTDIR}"/* 2>/dev/null || echo "(no files)"

    env > "${CONFIG_RESULTDIR}/env.txt"
done

###############################################################################
echo
echo "done! ------------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################

echo ">>>fin<<<"
