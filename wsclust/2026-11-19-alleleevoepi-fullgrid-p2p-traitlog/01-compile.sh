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
VENVDIR="${WORKDIR}/venv311"
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
echo "compile configs --------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
# allele-specific immunity epidemiology (allele-evoepi ABM port) with 2
# enabled sites, i.e., 4 strains, on full grid (subgrid 0, i.e., no subgrids),
# with p2p send probability 1 in 2,000 per cycle; four runs:
# - no p2p edges (no PE has a p2p target), default immunity waning (0.01)
# - random p2p edges (random PE pairs), default immunity waning (0.01)
# - control: random p2p edges, no host immune memory (waning rate 1.0, so
#   immunity gained on recovery is lost in the same generation)
# - control: random p2p edges, no immunity waning (waning rate 0.0)
# per-PE trait logger records the 3-bit trait value of each sampled host
# (0th population member), one multiplexed record per bit: bit 0 set if
# infected, and bits 1 and 2 the alleles at sites 0 and 1, for 5 possible values (0, or 1,
# 3, 5, 7 for the 4 strains)
# infection starts only on the corner PE (col 0, row 0), where all hosts start
# infected with the wildtype strain, and spreads from there by migration
# tournament size 2, so each host's update samples two random contacts, with
# an infected contact winning over an uninfected one
# note: globalSeed is u16 in kernel, and globalSeed * nRow * nCol is computed
# as u32, so seed must stay below 2**32 / (1170 * 755) ~= 4862 to avoid wrap
genome_flavor=genome_alleleevoepi_2026_11_19
subgrid=0
p2p=0.0005
seed=1118
for p2pModeWaning in none:0.01 pairs:0.01 pairs:1.0 pairs:0.0; do
    p2pMode="${p2pModeWaning%%:*}"
    waning="${p2pModeWaning##*:}"
    seed=$((seed + 1))
    CONFIG_NAME="flavor=${genome_flavor}+subgrid=${subgrid}+p2p=${p2p}+p2pmode=${p2pMode}+waning=${waning}+seed=${seed}"
    echo
    echo "====== Compiling config: ${CONFIG_NAME} ======"

    CONFIG_WORKDIR="${WORKDIR_STEP}/${CONFIG_NAME}"
    CONFIG_RESULTDIR="${RESULTDIR_STEP}/${CONFIG_NAME}"
    mkdir -p "${CONFIG_WORKDIR}"
    mkdir -p "${CONFIG_RESULTDIR}"

    ###########################################################################
    echo
    echo "config compile: ${CONFIG_NAME} ---------------------------------------"
    echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
    ###########################################################################
    cat > "${CONFIG_WORKDIR}/env.sh" << EOF
#!/usr/bin/env bash

export ASYNC_GA_ARCH_FLAG="wse3"
export ASYNC_GA_GENOME_FLAVOR="${genome_flavor}"
export ASYNC_GA_FABRIC_DIMS="762,1172"
export ASYNC_GA_NWAV=1
export ASYNC_GA_NCOL=755
export ASYNC_GA_NCOL_SUBGRID=${subgrid}
export ASYNC_GA_NONBLOCK=0
export ASYNC_GA_NROW=1170
export ASYNC_GA_NROW_SUBGRID=${subgrid}
export ASYNC_GA_NTRAIT=8
export ASYNC_GA_MSEC_AT_LEAST=0
export ASYNC_GA_NCYCLE_AT_LEAST=50000
export ASYNC_GA_POPSIZE=1133
export ASYNC_GA_POPULATION_FLAVOR="population_tournamentsteadystate"
export ASYNC_GA_TOURNSIZE_NUMERATOR=2
export ASYNC_GA_TOURNSIZE_DENOMINATOR=1
export ASYNC_GA_GLOBAL_SEED=${seed}
export COMPCONFENV_CEREBRASLIB_POPULATION_EXTINCTION_PROBABILITY__f32="0.0"
export COMPCONFENV_CEREBRASLIB_TRAITLOGGER_DSTREAM_ALGO_NAME__comptime_string="hybrid_0_steady_1_stretched_2_algo"
export COMPCONFENV_CEREBRASLIB_TRAITLOGGER_NUM_BITS__u32=1024
export COMPCONFENV_CEREBRASLIB_TRAITLOGGER_DILATION__u32=8
export COMPCONFENV_CEREBRASLIB_TRAITLOGGER_MULTIPLEX_LSB__u16=3
export COMPCONFENV_CEREBRASLIB_EVOEPI_N_SITES__u16=2
export COMPCONFENV_CEREBRASLIB_EVOEPI_WANING_RATE__f32="${waning}"
export COMPCONFENV_ASYNC_GA_P2P_SEND_PROBABILITY__f32="${p2p}"

echo "ASYNC_GA_ARCH_FLAG \${ASYNC_GA_ARCH_FLAG}"
echo "ASYNC_GA_GENOME_FLAVOR \${ASYNC_GA_GENOME_FLAVOR}"
echo "ASYNC_GA_FABRIC_DIMS \${ASYNC_GA_FABRIC_DIMS}"
echo "ASYNC_GA_NWAV \${ASYNC_GA_NWAV}"
echo "ASYNC_GA_NCOL \${ASYNC_GA_NCOL}"
echo "ASYNC_GA_NCOL_SUBGRID \${ASYNC_GA_NCOL_SUBGRID}"
echo "ASYNC_GA_NONBLOCK \${ASYNC_GA_NONBLOCK}"
echo "ASYNC_GA_NROW \${ASYNC_GA_NROW}"
echo "ASYNC_GA_NROW_SUBGRID \${ASYNC_GA_NROW_SUBGRID}"
echo "ASYNC_GA_NTRAIT \${ASYNC_GA_NTRAIT}"
echo "ASYNC_GA_MSEC_AT_LEAST \${ASYNC_GA_MSEC_AT_LEAST}"
echo "ASYNC_GA_NCYCLE_AT_LEAST \${ASYNC_GA_NCYCLE_AT_LEAST}"
echo "ASYNC_GA_POPSIZE \${ASYNC_GA_POPSIZE}"
echo "ASYNC_GA_POPULATION_FLAVOR \${ASYNC_GA_POPULATION_FLAVOR}"
echo "ASYNC_GA_TOURNSIZE_NUMERATOR \${ASYNC_GA_TOURNSIZE_NUMERATOR}"
echo "ASYNC_GA_TOURNSIZE_DENOMINATOR \${ASYNC_GA_TOURNSIZE_DENOMINATOR}"
echo "ASYNC_GA_GLOBAL_SEED \${ASYNC_GA_GLOBAL_SEED}"
echo "COMPCONFENV_CEREBRASLIB_POPULATION_EXTINCTION_PROBABILITY__f32 \${COMPCONFENV_CEREBRASLIB_POPULATION_EXTINCTION_PROBABILITY__f32}"
echo "COMPCONFENV_CEREBRASLIB_TRAITLOGGER_DSTREAM_ALGO_NAME__comptime_string \${COMPCONFENV_CEREBRASLIB_TRAITLOGGER_DSTREAM_ALGO_NAME__comptime_string}"
echo "COMPCONFENV_CEREBRASLIB_TRAITLOGGER_NUM_BITS__u32 \${COMPCONFENV_CEREBRASLIB_TRAITLOGGER_NUM_BITS__u32}"
echo "COMPCONFENV_CEREBRASLIB_TRAITLOGGER_DILATION__u32 \${COMPCONFENV_CEREBRASLIB_TRAITLOGGER_DILATION__u32}"
echo "COMPCONFENV_CEREBRASLIB_TRAITLOGGER_MULTIPLEX_LSB__u16 \${COMPCONFENV_CEREBRASLIB_TRAITLOGGER_MULTIPLEX_LSB__u16}"
echo "COMPCONFENV_CEREBRASLIB_EVOEPI_N_SITES__u16 \${COMPCONFENV_CEREBRASLIB_EVOEPI_N_SITES__u16}"
echo "COMPCONFENV_CEREBRASLIB_EVOEPI_WANING_RATE__f32 \${COMPCONFENV_CEREBRASLIB_EVOEPI_WANING_RATE__f32}"
echo "COMPCONFENV_ASYNC_GA_P2P_SEND_PROBABILITY__f32 \${COMPCONFENV_ASYNC_GA_P2P_SEND_PROBABILITY__f32}"
EOF

    chmod +x "${CONFIG_WORKDIR}/env.sh"
    cp "${CONFIG_WORKDIR}/env.sh" "${CONFIG_RESULTDIR}/env.sh"
    source "${CONFIG_WORKDIR}/env.sh"

    ###########################################################################
    echo
    echo "do compile: ${CONFIG_NAME} -------------------------------------------"
    echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
    ###########################################################################
    # runtime config sets p2p targets: none (no edges), or random pairs
    rm -rf "${WORKDIR}/src/kernel-async-ga/runtime_config_dispatch"
    echo "{\"dispatch_slug\": \"p2p-${p2pMode}\", \"p2pMode\": \"${p2pMode}\"}" \
        | tee "${CONFIG_WORKDIR}/runtime_config.jsonl"
    python3 "${WORKDIR}/src/kernel-async-ga/runtime_config_generate.py" \
        default.py.in "${CONFIG_WORKDIR}/runtime_config.jsonl"

    "${WORKDIR}/src/kernel-async-ga/compile.sh" | tee "${CONFIG_RESULTDIR}/compile.log"

    # Copy compiled artifacts to config workdir for run step
    cp -rL "${WORKDIR}/src/kernel-async-ga/cerebraslib" "${CONFIG_WORKDIR}/"
    cp -rL "${WORKDIR}/src/kernel-async-ga/out" "${CONFIG_WORKDIR}/"
    cp -L "${WORKDIR}/src/kernel-async-ga/client.py" "${CONFIG_WORKDIR}/"
    cp -rL "${WORKDIR}/src/kernel-async-ga/wheelhouse" "${CONFIG_WORKDIR}/"
    cp -L "${WORKDIR}/src/kernel-async-ga/compconf.json" "${CONFIG_WORKDIR}/"
    cp -rL "${WORKDIR}/src/kernel-async-ga/runtime_config_dispatch" "${CONFIG_WORKDIR}/"
    cp -rL "${WORKDIR}/src/kernel-async-ga/runtime_config_validate" "${CONFIG_WORKDIR}/"
    cp -rL "${CONFIG_WORKDIR}/runtime_config_dispatch" "${CONFIG_RESULTDIR}/"

    echo "Config ${CONFIG_NAME} compile complete!"
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
echo
echo "done! ------------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################

echo ">>>fin<<<"
