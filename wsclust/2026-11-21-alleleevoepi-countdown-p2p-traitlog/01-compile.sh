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
python3 -m uv pip freeze | tee "${RESULTDIR_STEP}/pip-freeze.txt" || :
python3 -m pylib_cs.cslc_wsclust_shim  # test install

###############################################################################
echo
echo "log source -------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
echo "log revision"
git -C "${FLOWDIR}" rev-parse HEAD > "${RESULTDIR_STEP}/git-revision.txt" || :
echo "log remote"
git -C "${FLOWDIR}" remote -v > "${RESULTDIR_STEP}/git-remote.txt" || :
echo "log status"
git -C "$(git -C "${FLOWDIR}" rev-parse --show-toplevel)" status \
    > "${RESULTDIR_STEP}/git-status.txt" || :
echo "log diff"
git -C "${FLOWDIR}" --no-pager diff > "${RESULTDIR_STEP}/git-status.diff" || :
git -C "${FLOWDIR}" ls-files -z --others --exclude-standard --exclude=resultdir --exclude=workdir | xargs -0 -I {} git -C "${FLOWDIR}" --no-pager diff --no-index /dev/null {} >> "${RESULTDIR_STEP}/git-status.diff" || :

SRCDIR="${WORKDIR}/src"
echo "SRCDIR ${SRCDIR}"
echo "log revision"
git -C "${SRCDIR}" rev-parse HEAD > "${RESULTDIR_STEP}/src-revision.txt" || :
echo "log remote"
git -C "${SRCDIR}" remote -v > "${RESULTDIR_STEP}/src-remote.txt" || :
echo "log status"
git -C "$(git -C "${SRCDIR}" rev-parse --show-toplevel)" status \
    > "${RESULTDIR_STEP}/src-status.txt" || :
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
# enabled sites, i.e., 4 strains, with nibble countdowns for infection and
# immunity durations (genome_alleleevoepi_2026_11_21): infection lasts exactly
# 10 generations (countdown cap 10, step probability 1), and immunity to each
# allele a sum of 15 geometric waits (countdown cap 15), mean 100 or 500
# generations, renewed on reinfection's recovery; within-host evolution is
# neutral (WITHINHOST_B 0), with WITHINHOST_T 1, so each site of a new
# infection mutates with probability equal to the mutation rate, regardless
# of host immunity (i.e., mutation rate is per site per transmission)
# p2p edges are random PE pairs (within subgrids, if any), one genome per p2p
# message, and only infected hosts are sent
# seven independent runs (one job each), each set by its own runtime config
# (p2p targets, and per run p2p send probability, mutation rate, seed), from
# two compiles:
# 1. full grid, as 2026-11-20 workflow's baseline (no subgrids, mean immunity
#    100 generations, mutation rate 1e-6, 1 genome per neighbor send, recv
#    batches of 4, population 6800, infection seeded on corner PE only), but
#    p2p send probability 1 in 20
# 2-7. nested subgrids on the 1152 x 576 PEs that full (non-overflowing)
#    subgrids occupy (runtime config nestedSubgridSizeFactor 2): 9 x 9 PE
#    subgrids at the corner PE (col 0, row 0), then L-shaped shells of 3
#    subgrids each of 18, 36, 72, 144, and 288 PEs on a side, and one 576 x
#    576 subgrid; sweeping mutation rate 2^-16 to 2^-26 by factors of 4
#    (named runtime config rtc_evoepi_mutationRate); p2p send probability 1
#    in 50; migration between neighbors 10x the default (10 genomes per send,
#    received in batches of 40); mean immunity 500 generations; infection
#    seeded on every PE whose col and row are multiples of 9 (seed PE stride
#    9); population 6500, below 6649 limit of PE memory with 10x neighbor
#    buffers
# per-PE trait logger records the 3-bit trait value of each sampled host
# (0th population member), one multiplexed record per bit: bit 0 set if
# infected, and bits 1 and 2 the alleles at sites 0 and 1
# tournament size 2, so each host's update samples two random contacts, with
# an infected contact winning over an uninfected one
# note: globalSeed is u16 in kernel, and globalSeed * nRow * nCol is computed
# as u32, so seed must stay below 2**32 / (1170 * 755) ~= 4862 to avoid wrap
genome_flavor=genome_alleleevoepi_2026_11_21
p2pMode=pairs

# compile, for each compile config; and write each run config's runtime
# config, with compiled artifacts, to its own directory (one job per run
# config in run step)
for compileConfig in fullgrid nested; do
    if [ "${compileConfig}" = "fullgrid" ]; then
        nrow=1170
        ncol=755
        p2p=0.05  # compiled-in, runtime config sets it too
        immunityp=0.15  # countdown cap 15: mean 100 generations
        sendgenomes=1
        recvgenomes=4
        subgrid=0
        seedstride=1170  # only corner PE
        popsize=6800  # below 6975 limit of PE memory
        compileseed=1130  # runtime config sets it too
    else
        nrow=1152  # only PEs of full (non-overflowing) nested subgrids
        ncol=576
        p2p=0.02
        immunityp=0.03  # countdown cap 15: mean 500 generations
        sendgenomes=10
        recvgenomes=40
        subgrid=9
        seedstride=9
        popsize=6500  # below 6649 limit with 10x neighbor buffers
        compileseed=1131  # runtime config overrides, per run
    fi
    COMPILE_WORKDIR="${WORKDIR_STEP}/compile=${compileConfig}"
    COMPILE_RESULTDIR="${RESULTDIR_STEP}/compile=${compileConfig}"
    mkdir -p "${COMPILE_WORKDIR}"
    mkdir -p "${COMPILE_RESULTDIR}"

    ###########################################################################
    echo
    echo "compile: ${compileConfig} --------------------------------------------"
    echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
    ###########################################################################
    cat > "${COMPILE_WORKDIR}/env.sh" << EOF
#!/usr/bin/env bash

export ASYNC_GA_ARCH_FLAG="wse3"
export ASYNC_GA_GENOME_FLAVOR="${genome_flavor}"
export ASYNC_GA_FABRIC_DIMS="762,1172"
export ASYNC_GA_NWAV=1
export ASYNC_GA_NCOL=${ncol}
export ASYNC_GA_NCOL_SUBGRID=${subgrid}
export ASYNC_GA_NONBLOCK=0
export ASYNC_GA_NROW=${nrow}
export ASYNC_GA_NROW_SUBGRID=${subgrid}
export ASYNC_GA_NTRAIT=8
export ASYNC_GA_MSEC_AT_LEAST=0
export ASYNC_GA_NCYCLE_AT_LEAST=50000
export ASYNC_GA_POPSIZE=${popsize}
export ASYNC_GA_POPULATION_FLAVOR="population_tournamentsteadystate"
export ASYNC_GA_TOURNSIZE_NUMERATOR=2
export ASYNC_GA_TOURNSIZE_DENOMINATOR=1
export ASYNC_GA_GLOBAL_SEED=${compileseed}
export COMPCONFENV_CEREBRASLIB_POPULATION_EXTINCTION_PROBABILITY__f32="0.0"
export COMPCONFENV_CEREBRASLIB_TRAITLOGGER_DSTREAM_ALGO_NAME__comptime_string="hybrid_0_steady_1_stretched_2_algo"
export COMPCONFENV_CEREBRASLIB_TRAITLOGGER_NUM_BITS__u32=1024
export COMPCONFENV_CEREBRASLIB_TRAITLOGGER_DILATION__u32=1  # no dilation: sample every cycle
export COMPCONFENV_CEREBRASLIB_TRAITLOGGER_MULTIPLEX_LSB__u16=3
export COMPCONFENV_CEREBRASLIB_EVOEPI_N_SITES__u16=2
export COMPCONFENV_CEREBRASLIB_EVOEPI_SEED_PE_STRIDE__u16=${seedstride}
export COMPCONFENV_CEREBRASLIB_EVOEPI_INFECTION_COUNTDOWN_CAP__u16=10
export COMPCONFENV_CEREBRASLIB_EVOEPI_INFECTION_COUNTDOWN_P__f32="1.0"
export COMPCONFENV_CEREBRASLIB_EVOEPI_IMMUNITY_COUNTDOWN_CAP__u16=15
export COMPCONFENV_CEREBRASLIB_EVOEPI_IMMUNITY_COUNTDOWN_P__f32="${immunityp}"
export COMPCONFENV_CEREBRASLIB_EVOEPI_WITHINHOST_B__f32="0.0"
export COMPCONFENV_CEREBRASLIB_EVOEPI_WITHINHOST_T__f32="1.0"
export COMPCONFENV_CEREBRASLIB_EVOEPI_MUTATION_RATE__f32="1e-6"  # runtime config overrides
export COMPCONFENV_ASYNC_GA_P2P_SEND_PROBABILITY__f32="${p2p}"
export COMPCONFENV_ASYNC_GA_SEND_NUM_GENOMES__u16=${sendgenomes}
export COMPCONFENV_ASYNC_GA_RECV_NUM_GENOMES__u16=${recvgenomes}

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
echo "COMPCONFENV_CEREBRASLIB_EVOEPI_SEED_PE_STRIDE__u16 \${COMPCONFENV_CEREBRASLIB_EVOEPI_SEED_PE_STRIDE__u16}"
echo "COMPCONFENV_CEREBRASLIB_EVOEPI_INFECTION_COUNTDOWN_CAP__u16 \${COMPCONFENV_CEREBRASLIB_EVOEPI_INFECTION_COUNTDOWN_CAP__u16}"
echo "COMPCONFENV_CEREBRASLIB_EVOEPI_INFECTION_COUNTDOWN_P__f32 \${COMPCONFENV_CEREBRASLIB_EVOEPI_INFECTION_COUNTDOWN_P__f32}"
echo "COMPCONFENV_CEREBRASLIB_EVOEPI_IMMUNITY_COUNTDOWN_CAP__u16 \${COMPCONFENV_CEREBRASLIB_EVOEPI_IMMUNITY_COUNTDOWN_CAP__u16}"
echo "COMPCONFENV_CEREBRASLIB_EVOEPI_IMMUNITY_COUNTDOWN_P__f32 \${COMPCONFENV_CEREBRASLIB_EVOEPI_IMMUNITY_COUNTDOWN_P__f32}"
echo "COMPCONFENV_CEREBRASLIB_EVOEPI_WITHINHOST_B__f32 \${COMPCONFENV_CEREBRASLIB_EVOEPI_WITHINHOST_B__f32}"
echo "COMPCONFENV_CEREBRASLIB_EVOEPI_WITHINHOST_T__f32 \${COMPCONFENV_CEREBRASLIB_EVOEPI_WITHINHOST_T__f32}"
echo "COMPCONFENV_CEREBRASLIB_EVOEPI_MUTATION_RATE__f32 \${COMPCONFENV_CEREBRASLIB_EVOEPI_MUTATION_RATE__f32}"
echo "COMPCONFENV_ASYNC_GA_P2P_SEND_PROBABILITY__f32 \${COMPCONFENV_ASYNC_GA_P2P_SEND_PROBABILITY__f32}"
echo "COMPCONFENV_ASYNC_GA_SEND_NUM_GENOMES__u16 \${COMPCONFENV_ASYNC_GA_SEND_NUM_GENOMES__u16}"
echo "COMPCONFENV_ASYNC_GA_RECV_NUM_GENOMES__u16 \${COMPCONFENV_ASYNC_GA_RECV_NUM_GENOMES__u16}"
EOF
    chmod +x "${COMPILE_WORKDIR}/env.sh"
    cp "${COMPILE_WORKDIR}/env.sh" "${COMPILE_RESULTDIR}/env.sh"
    source "${COMPILE_WORKDIR}/env.sh"

    "${WORKDIR}/src/kernel-async-ga/compile.sh" | tee "${COMPILE_RESULTDIR}/compile.log"
    rm -rf "${COMPILE_WORKDIR}/out"
    cp -rL "${WORKDIR}/src/kernel-async-ga/out" "${COMPILE_WORKDIR}/"
    cp -L "${WORKDIR}/src/kernel-async-ga/compconf.json" "${COMPILE_WORKDIR}/"

    # run configs, one job each, as name:p2p:mutexp:seed, for runtime p2p
    # send probability, mutation rate 2^-mutexp, and seed
    if [ "${compileConfig}" = "fullgrid" ]; then
        RUN_CONFIGS="fullgrid-p2p0.05:0.05:none:1130"
        nest=""
    else
        RUN_CONFIGS="nested-mutlog2-16:0.02:16:1131 nested-mutlog2-18:0.02:18:1132 nested-mutlog2-20:0.02:20:1133 nested-mutlog2-22:0.02:22:1134 nested-mutlog2-24:0.02:24:1135 nested-mutlog2-26:0.02:26:1136"
        nest=", \"nestedSubgridSizeFactor\": 2"
    fi
    for runConfig in ${RUN_CONFIGS}; do
        IFS=: read -r runName runP2p runMutexp runSeed <<< "${runConfig}"
        CONFIG_NAME="flavor=${genome_flavor}+run=${runName}+subgrid=${subgrid}+immunityp=${immunityp}+sendgenomes=${sendgenomes}+seed=${runSeed}"
        echo
        echo "====== Configuring run: ${CONFIG_NAME} ======"

        CONFIG_WORKDIR="${WORKDIR_STEP}/${CONFIG_NAME}"
        CONFIG_RESULTDIR="${RESULTDIR_STEP}/${CONFIG_NAME}"
        mkdir -p "${CONFIG_WORKDIR}"
        mkdir -p "${CONFIG_RESULTDIR}"

        #######################################################################
        echo
        echo "runtime config: ${CONFIG_NAME} ---------------------------------"
        echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
        #######################################################################
        # p2p targets (random pairs, within subgrids if any), nested subgrid
        # sizes, p2p send probability, mutation rate, and seed
        rm -rf "${WORKDIR}/src/kernel-async-ga/runtime_config_dispatch"
        if [ "${runMutexp}" = "none" ]; then
            mutation=1e-6
        else
            mutation="$(python3 -c "print(repr(2.0 ** -${runMutexp}))")"  # exact f32
        fi
        echo "{\"dispatch_slug\": \"${runName}\", \"p2pMode\": \"${p2pMode}\"${nest}, \"p2pSendProbability\": ${runP2p}, \"globalSeed\": ${runSeed}, \"rtc_evoepi_mutationRate:f32\": ${mutation}}" \
            | tee "${CONFIG_WORKDIR}/runtime_config.jsonl"
        cp "${CONFIG_WORKDIR}/runtime_config.jsonl" "${CONFIG_RESULTDIR}/"
        python3 "${WORKDIR}/src/kernel-async-ga/runtime_config_generate.py" \
            default.py.in "${CONFIG_WORKDIR}/runtime_config.jsonl"

        # copy compiled artifacts to config workdir for run step
        cp -L "${COMPILE_WORKDIR}/env.sh" "${CONFIG_WORKDIR}/"
        cp -L "${COMPILE_WORKDIR}/env.sh" "${CONFIG_RESULTDIR}/"
        cp -rL "${WORKDIR}/src/kernel-async-ga/cerebraslib" "${CONFIG_WORKDIR}/"
        cp -rL "${COMPILE_WORKDIR}/out" "${CONFIG_WORKDIR}/"
        cp -L "${WORKDIR}/src/kernel-async-ga/client.py" "${CONFIG_WORKDIR}/"
        cp -rL "${WORKDIR}/src/kernel-async-ga/wheelhouse" "${CONFIG_WORKDIR}/"
        cp -L "${COMPILE_WORKDIR}/compconf.json" "${CONFIG_WORKDIR}/"
        cp -rL "${WORKDIR}/src/kernel-async-ga/runtime_config_dispatch" "${CONFIG_WORKDIR}/"
        cp -rL "${WORKDIR}/src/kernel-async-ga/runtime_config_validate" "${CONFIG_WORKDIR}/"
        cp -rL "${CONFIG_WORKDIR}/runtime_config_dispatch" "${CONFIG_RESULTDIR}/"

        echo "Config ${CONFIG_NAME} setup complete!"
    done
done

###############################################################################
echo
echo "closeout ---------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################
find "${RESULTDIR_STEP}" | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/" || :
du -ah "${RESULTDIR_STEP}"/* || :

env > "${RESULTDIR_STEP}/env.txt" || :

###############################################################################
echo
echo "done! ------------------------------------------------------------------"
echo ">>>>> ${FLOWNAME} :: ${STEPNAME} || ${SECONDS}"
###############################################################################

echo ">>>fin<<<"
