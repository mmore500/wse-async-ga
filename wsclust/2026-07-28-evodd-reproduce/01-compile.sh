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
echo "setup venv -------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
VENVDIR="${WORKDIR}/venv38"
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
git -C "${FLOWDIR}" ls-files -z --others --exclude-standard --exclude=resultdir --exclude=workdir | xargs -0 -I {} git -C "${FLOWDIR}" --no-pager diff --no-index /dev/null {} >> "${RESULTDIR_STEP}/git-status.diff" || :

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
echo "setup compile ----------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
export CSLC="${CSLC:-cslc}"
echo "CSLC ${CSLC}"
export ASYNC_GA_FABRIC_DIMS="762,1172"
echo "ASYNC_GA_FABRIC_DIMS ${ASYNC_GA_FABRIC_DIMS}"

WORKDIR_MNT="$(realpath path)"
WORKDIR_MNT="${WORKDIR_MNT#/}"
WORKDIR_MNT="${WORKDIR_MNT%%/*}"
echo "WORKDIR_MNT ${WORKDIR_MNT}"
export SINGULARITY_BIND="${SINGULARITY_BIND:+${SINGULARITY_BIND},}/${WORKDIR_MNT}:/${WORKDIR_MNT},/tmp:/tmp"
echo "SINGULARITY_BIND ${SINGULARITY_BIND}"

###############################################################################
echo
echo "config compile ---------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
# genome_evodd (a 141-word/gSize genome implementing the Kouvaris et al. 2017
# GRN development model from
# https://github.com/mmore500/evodd/blob/main/bindle/2026-07-23-exploratory.py)
# MUST run with population_tournamentsteadystate, not
# population_tournamentgenerational: the generational flavor's bulk
# selection copy uses a DSD stride computed from genome.gSize that is
# hardware-limited to fit in an i8 (max 127 words); genome_evodd's gSize=141
# overflows that limit and fails to compile *at any population size*
# (confirmed empirically: `cslc` rejects it with "integer value '141'
# cannot be coerced to type 'i8'" even at popSize=1). The steadystate
# flavor instead does its migrant-swap/selection copies with plain scalar
# loops, sidestepping the stride limit, and also drops the old flavor's
# second (`swap`) population buffer and fitBuf/indexBuf scratch arrays,
# roughly halving per-PE population memory -- which is what makes a
# 141-word genome affordable at all.
export ASYNC_GA_POPULATION_FLAVOR="population_tournamentsteadystate"
echo "ASYNC_GA_POPULATION_FLAVOR ${ASYNC_GA_POPULATION_FLAVOR}"

cat > "${WORKDIR_STEP}/env.sh" << 'EOF'
#!/usr/bin/env bash

export ASYNC_GA_ARCH_FLAG="wse3"
export ASYNC_GA_GENOME_FLAVOR="genome_evodd"
export ASYNC_GA_POPULATION_FLAVOR="population_tournamentsteadystate"
export ASYNC_GA_FABRIC_DIMS="762,1172"
export ASYNC_GA_NWAV=141
export ASYNC_GA_NCOL=755
export ASYNC_GA_NCOL_SUBGRID=0
export ASYNC_GA_NONBLOCK=0
export ASYNC_GA_NROW=1170
export ASYNC_GA_NROW_SUBGRID=0
export ASYNC_GA_NTRAIT=1
export ASYNC_GA_MSEC_AT_LEAST=0
# 1,000,000 self-activation cycles == 1,000,000 generations per PE. Measured
# locally (CS fabric simulator device-clock/tsc telemetry, a=perf+...pqt)
# for genome_evodd + population_tournamentsteadystate at popSize=22 on a
# small grid: a fully-connected (4-neighbor, i.e. interior-tile-like) PE
# sustains ~2300-2900 generations/sec of 850MHz-equivalent real hardware
# time (~330-430us/generation; less-connected corner/edge PEs in the small
# test grid ran faster, up to ~3900/sec, so 2300/sec is the conservative
# figure representative of a large grid's interior tiles). At that rate,
# 1,000,000 cycles is ~6-7 minutes of real device time and covers ~330
# full schedule periods (each period is
# 3 * CEREBRASLIB_EVODD_SCHEDULE_BLOCKLEN generations; default block length
# is 1000, so 3000 generations/period) -- ample for the switching-pattern
# GRN benefit to converge, echoing the original numpy notebook/paper's
# many-generation training runs. Tune up/down as needed; this genome/
# population/grid combo does ~2300-2900 generations per real hardware
# second, so ~700k-870k generations are achievable in 5 minutes of device
# time.
export ASYNC_GA_NCYCLE_AT_LEAST=1000000
# popSize=22 is the empirically-determined maximum that compiles for
# genome_evodd + population_tournamentsteadystate on wse3: binary-searched
# locally via `cslc` (popSize=22 succeeds, popSize=23 fails with "ran out
# of PE memory for task table"/"data (section .data.hi)"). This bound is a
# property of a single PE's fixed local SBUF budget (population buffer is
# popSize * 141 words, plus send/recv buffers, hstrat/traitlogger storage,
# task table, etc.) and does not depend on fabric-dims/grid size, so it
# should hold at this grid too -- re-verify with a quick compile-only dry
# run before committing cluster time if genome_evodd's memory footprint
# changes.
export ASYNC_GA_POPSIZE=22
export ASYNC_GA_TOURNSIZE_NUMERATOR=5
export ASYNC_GA_TOURNSIZE_DENOMINATOR=1
export ASYNC_GA_GLOBAL_SEED=1
export COMPCONFENV_CEREBRASLIB_HSTRAT_DSTREAM_ALGO_NAME__comptime_string="hybrid_0_steady_2_tilted_3_algo"
export COMPCONFENV_CEREBRASLIB_HSTRAT_T_DILATION__u32=1
# genome_evodd defaults (see cerebraslib/genome/genome_evodd.csl); spelled
# out here for visibility/tunability rather than left implicit.
export COMPCONFENV_CEREBRASLIB_EVODD_SCHEDULE_BLOCKLEN__u32=1000
export COMPCONFENV_CEREBRASLIB_EVODD_L1_SCALE__f32=1.0
export COMPCONFENV_CEREBRASLIB_EVODD_L2_SCALE__f32=0.0
export COMPCONFENV_CEREBRASLIB_EVODD_LAM1__f32=0.22
export COMPCONFENV_CEREBRASLIB_EVODD_LAM2__f32=38.0

echo "ASYNC_GA_ARCH_FLAG ${ASYNC_GA_ARCH_FLAG}"
echo "ASYNC_GA_GENOME_FLAVOR ${ASYNC_GA_GENOME_FLAVOR}"
echo "ASYNC_GA_POPULATION_FLAVOR ${ASYNC_GA_POPULATION_FLAVOR}"
echo "ASYNC_GA_FABRIC_DIMS ${ASYNC_GA_FABRIC_DIMS}"
echo "ASYNC_GA_NWAV ${ASYNC_GA_NWAV}"
echo "ASYNC_GA_NCOL ${ASYNC_GA_NCOL}"
echo "ASYNC_GA_NCOL_SUBGRID ${ASYNC_GA_NCOL_SUBGRID}"
echo "ASYNC_GA_NONBLOCK ${ASYNC_GA_NONBLOCK}"
echo "ASYNC_GA_NROW ${ASYNC_GA_NROW}"
echo "ASYNC_GA_NROW_SUBGRID ${ASYNC_GA_NROW_SUBGRID}"
echo "ASYNC_GA_NTRAIT ${ASYNC_GA_NTRAIT}"
echo "ASYNC_GA_MSEC_AT_LEAST ${ASYNC_GA_MSEC_AT_LEAST}"
echo "ASYNC_GA_NCYCLE_AT_LEAST ${ASYNC_GA_NCYCLE_AT_LEAST}"
echo "ASYNC_GA_POPSIZE ${ASYNC_GA_POPSIZE}"
echo "ASYNC_GA_TOURNSIZE_NUMERATOR ${ASYNC_GA_TOURNSIZE_NUMERATOR}"
echo "ASYNC_GA_TOURNSIZE_DENOMINATOR ${ASYNC_GA_TOURNSIZE_DENOMINATOR}"
echo "ASYNC_GA_GLOBAL_SEED ${ASYNC_GA_GLOBAL_SEED}"
echo "COMPCONFENV_CEREBRASLIB_HSTRAT_DSTREAM_ALGO_NAME__comptime_string ${COMPCONFENV_CEREBRASLIB_HSTRAT_DSTREAM_ALGO_NAME__comptime_string}"
echo "COMPCONFENV_CEREBRASLIB_HSTRAT_T_DILATION__u32 ${COMPCONFENV_CEREBRASLIB_HSTRAT_T_DILATION__u32}"
echo "COMPCONFENV_CEREBRASLIB_EVODD_SCHEDULE_BLOCKLEN__u32 ${COMPCONFENV_CEREBRASLIB_EVODD_SCHEDULE_BLOCKLEN__u32}"
echo "COMPCONFENV_CEREBRASLIB_EVODD_L1_SCALE__f32 ${COMPCONFENV_CEREBRASLIB_EVODD_L1_SCALE__f32}"
echo "COMPCONFENV_CEREBRASLIB_EVODD_L2_SCALE__f32 ${COMPCONFENV_CEREBRASLIB_EVODD_L2_SCALE__f32}"
echo "COMPCONFENV_CEREBRASLIB_EVODD_LAM1__f32 ${COMPCONFENV_CEREBRASLIB_EVODD_LAM1__f32}"
echo "COMPCONFENV_CEREBRASLIB_EVODD_LAM2__f32 ${COMPCONFENV_CEREBRASLIB_EVODD_LAM2__f32}"
EOF

chmod +x "${WORKDIR_STEP}/env.sh"
cp "${WORKDIR_STEP}/env.sh" "${RESULTDIR_STEP}/env.sh"
source "${WORKDIR_STEP}/env.sh"

###############################################################################
echo
echo "do compile ---------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
"${WORKDIR}/src/kernel-async-ga/compile.sh" | tee "${RESULTDIR_STEP}/compile.log"

# Copy compiled artifacts to workdir for run step
cp -rL "${WORKDIR}/src/kernel-async-ga/cerebraslib" "${WORKDIR_STEP}/"
cp -rL "${WORKDIR}/src/kernel-async-ga/out" "${WORKDIR_STEP}/"
cp -L "${WORKDIR}/src/kernel-async-ga/client.py" "${WORKDIR_STEP}/"
cp -L "${WORKDIR}/src/kernel-async-ga/compconf.json" "${WORKDIR_STEP}/"

###############################################################################
echo
echo "closeout ---------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
find "${RESULTDIR_STEP}" | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/"
du -ah "${RESULTDIR_STEP}"/*

env > "${RESULTDIR_STEP}/env.txt"

###############################################################################
echo
echo "done! ------------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################

echo ">>>fin<<<"
