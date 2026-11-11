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
echo "setup venv ---------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
VENVDIR="${WORKDIR}/venv3x"
echo "VENVDIR ${VENVDIR}"

echo "creating venv"
source "${VENVDIR}/bin/activate"
python3 -m uv pip freeze | tee "${RESULTDIR_STEP}/pip-freeze.txt"

###############################################################################
echo
echo "summarize run ------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
WORKDIR_RUN="${WORKDIR}/02-run/out"
echo "WORKDIR_RUN ${WORKDIR_RUN}"

python3 - << EOF
import glob

import polars as pl

# fitness/genome summary --------------------------------------------------
# "a=genomes+..." carries one row per PE, sampling that PE's 0th-slot
# population member (fitness column) at halt. genome_evodd's fitness is
# benefit-of-GRN-response-to-current-training-pattern minus an L1/L2
# regularization term on the dev matrix; a freshly-initialized (all-zero
# dev matrix) genome scores ~0.5 (benefit only, no learned response) with
# zero regularization cost, so a population that has evolved a useful
# switching-pattern GRN should show mean fitness meaningfully above 0.5 --
# this is the local, per-tile echo of the original numpy notebook/paper's
# "benefit improves as the GRN evolves" result.
genome_paths = sorted(glob.glob("${WORKDIR_RUN}/a=genomes+*.pqt"))
print(f"genome_paths={genome_paths}")
if genome_paths:
    df = pl.read_parquet(genome_paths[-1])
    fitness = df["fitness"]
    print("fitness summary ==========================================")
    print(f"n={fitness.len()}")
    print(f"mean={fitness.mean():.4f}")
    print(f"median={fitness.median():.4f}")
    print(f"std={fitness.std():.4f}")
    print(f"min={fitness.min():.4f}")
    print(f"max={fitness.max():.4f}")
    baseline = 0.5
    frac_above_baseline = (fitness > baseline).mean()
    print(
        f"fraction of PEs with fitness above untrained baseline "
        f"({baseline}): {frac_above_baseline:.2%}"
    )
else:
    print("no a=genomes+*.pqt found -- was the run step's --no-process-fossils"
          " client.py invocation successful?")

# device-clock throughput summary -----------------------------------------
perf_paths = sorted(glob.glob("${WORKDIR_RUN}/a=perf+*.pqt"))
print(f"perf_paths={perf_paths}")
if perf_paths:
    df = pl.read_parquet(perf_paths[-1])
    hz = df["tsc cycle hertz"]
    ncycle = df["nCycle"][0]
    print("device-clock (850MHz-equivalent) throughput summary =====")
    print(f"mean generations/sec={hz.mean():.1f}")
    print(f"median generations/sec={hz.median():.1f}")
    print(f"min generations/sec={hz.min():.1f} (most-connected/slowest PE)")
    print(f"max generations/sec={hz.max():.1f} (least-connected/fastest PE)")
    print(f"nCycleAtLeast (generations run)={ncycle}")
    print(
        "generations achievable in 5 real hardware minutes "
        f"(median rate)={hz.median() * 300:.0f}"
    )
else:
    print("no a=perf+*.pqt found")
EOF

###############################################################################
echo
echo "closeout ---------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
cp "${WORKDIR_RUN}"/a=genomes+*.pqt "${RESULTDIR_STEP}/" 2>/dev/null || :
cp "${WORKDIR_RUN}"/a=perf+*.pqt "${RESULTDIR_STEP}/" 2>/dev/null || :
cp "${WORKDIR_RUN}"/a=traits+*.pqt "${RESULTDIR_STEP}/" 2>/dev/null || :

find "${RESULTDIR_STEP}" | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/"
du -ah "${RESULTDIR_STEP}"/* 2>/dev/null || echo "(no files)"

env > "${RESULTDIR_STEP}/env.txt"

###############################################################################
echo
echo "done! ------------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################

echo ">>>fin<<<"
