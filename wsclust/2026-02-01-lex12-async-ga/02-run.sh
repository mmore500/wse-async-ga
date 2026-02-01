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
VENVDIR="${WORKDIR}/venv"
echo "VENVDIR ${VENVDIR}"

echo "creating venv"
source "${VENVDIR}/bin/activate"
python3 -m uv pip freeze | tee "${RESULTDIR_STEP}/pip-freeze.txt"
python3 -m pylib_cs.cslc_wsclust_shim  # test install

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
echo "run configs ------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
# lex12 configuration: two genome flavors with seeds 1 and 2
seed=0
for genome_flavor in genome_purifyingonly genome_purifyingplus; do
    seed=$((seed + 1))
    CONFIG_NAME="flavor=${genome_flavor}+seed=${seed}"
    echo
    echo "====== Running config: ${CONFIG_NAME} ======"

    CONFIG_COMPILE_WORKDIR="${WORKDIR}/01-compile/${CONFIG_NAME}"
    CONFIG_WORKDIR="${WORKDIR_STEP}/${CONFIG_NAME}"
    CONFIG_RESULTDIR="${RESULTDIR_STEP}/${CONFIG_NAME}"
    mkdir -p "${CONFIG_WORKDIR}"
    mkdir -p "${CONFIG_RESULTDIR}"

    ###########################################################################
    echo
    echo "setup run: ${CONFIG_NAME} --------------------------------------------"
    echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
    ###########################################################################
    source "${CONFIG_COMPILE_WORKDIR}/env.sh"

    export ASYNC_GA_MAX_FOSSIL_SETS=2500
    echo "ASYNC_GA_MAX_FOSSIL_SETS=${ASYNC_GA_MAX_FOSSIL_SETS}"

    mkdir -p "${CONFIG_WORKDIR}/out"
    mkdir -p "${CONFIG_WORKDIR}/run"

    cd "${CONFIG_WORKDIR}/run"
    echo "PWD ${PWD}"
    ls
    echo "copying kernel-async-ga files from ${CONFIG_COMPILE_WORKDIR}..."
    cp -rL "${CONFIG_COMPILE_WORKDIR}/cerebraslib" .
    cp -rL "${CONFIG_COMPILE_WORKDIR}/out" .
    cp -L "${CONFIG_COMPILE_WORKDIR}/client.py" .
    cp -L "${CONFIG_COMPILE_WORKDIR}/compconf.json" .
    ls
    cat client.py

    cd "${CONFIG_WORKDIR}"
    echo "PWD ${PWD}"
    find "./run" | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/"

    ###########################################################################
    echo
    echo "do run: ${CONFIG_NAME} -----------------------------------------------"
    echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
    ###########################################################################
    cd "${CONFIG_WORKDIR}"
    echo "PWD ${PWD}"

    which python3
    python3 - << EOF
import logging
import os

from cerebras.appliance import logger
from cerebras.sdk.client import SdkLauncher


# Enable DEBUG level logging for more telemetry
logging.basicConfig(
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.DEBUG,
)

# Also set appliance logger to DEBUG
logger.setLevel(logging.DEBUG)

logging.info("entering SdkLauncher")
with SdkLauncher("./run", disable_version_check=True) as launcher:

    logging.info("querying context info...")
    response = launcher.run(
        "env",
        "pwd",
        "ls",
        r'find . | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/"',
    )
    logging.info("... done!")
    logging.info(response + "\n")

    env_prefix = " ".join(
        f"{key}='{value}'"
        for key, value in os.environ.items()
        if key.startswith("ASYNC_GA_") or key.startswith("COMPCONFENV_")
    )
    command = f"{env_prefix} cs_python client.py --cmaddr %CMADDR% 2>&1 | tee run.log"
    logging.info(f"command={command}")
    logging.info("running command...")
    response = launcher.run(command)
    logging.info("... done!")
    logging.info(response + "\n")

    logging.info("capturing WSJOB_ID...")
    wsjob_id = launcher.run("echo $WSJOB_ID").strip()
    logging.info(f"WSJOB_ID={wsjob_id}")
    with open("${CONFIG_WORKDIR}/out/wsjob_id.txt", "w") as f:
        f.write(wsjob_id + "\n")
    logging.info("... done!")

    logging.info("finding output files...")
    response = launcher.run(
        "find . -maxdepth 1 -type f "
        r'\( -name "*.log" -o -name "*.pqt" -o -name "*.json" -o -name "*.npy" \)',
    )
    logging.info("... done!")
    logging.info(response + "\n")

    for filename in response.splitlines():
        target = f"${CONFIG_WORKDIR}/out/{filename}"
        logging.info(f"retrieving file {filename} to {target}...")
        file_contents = launcher.download_artifact(filename, target)
        logging.info("... done!")

    logging.info("exiting SdkLauncher")

logging.info("exited SdkLauncher")
EOF

    ###########################################################################
    echo
    echo "get csctl job info: ${CONFIG_NAME} ----------------------------------"
    echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
    ###########################################################################
    WSJOB_ID_FILE="${CONFIG_WORKDIR}/out/wsjob_id.txt"
    if [ -f "${WSJOB_ID_FILE}" ]; then
        WSJOB_ID="$(cat "${WSJOB_ID_FILE}")"
        echo "WSJOB_ID ${WSJOB_ID}"
        if [ -n "${WSJOB_ID}" ]; then
            echo "running csctl get job ${WSJOB_ID}..."
            csctl get job "${WSJOB_ID}" -oyaml > "${CONFIG_WORKDIR}/out/csctl-job.yaml" 2>&1 || :
            echo "running csctl get job ${WSJOB_ID} events..."
            csctl get job "${WSJOB_ID}" --show-events -oyaml > "${CONFIG_WORKDIR}/out/csctl-job-events.yaml" 2>&1 || :
            echo "... done!"
        else
            echo "WSJOB_ID is empty, skipping csctl"
        fi
    else
        echo "WSJOB_ID file not found: ${WSJOB_ID_FILE}"
    fi

    ###########################################################################
    echo
    echo "closeout: ${CONFIG_NAME} ---------------------------------------------"
    echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
    ###########################################################################
    find "${CONFIG_WORKDIR}/out" | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/"
    du -ah "${CONFIG_WORKDIR}"/out/*

    cp "${CONFIG_WORKDIR}"/out/* "${CONFIG_RESULTDIR}"

    find "${CONFIG_RESULTDIR}" | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/"
    du -ah "${CONFIG_RESULTDIR}"/*

    env > "${CONFIG_RESULTDIR}/env.txt"

    echo "Config ${CONFIG_NAME} run complete!"
done

###############################################################################
echo
echo "closeout ---------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
find "${RESULTDIR_STEP}" | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/"
du -ah "${RESULTDIR_STEP}"/*

env > "${RESULTDIR_STEP}/env.txt"

###############################################################################
echo "done! ------------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################

echo ">>>fin<<<"
