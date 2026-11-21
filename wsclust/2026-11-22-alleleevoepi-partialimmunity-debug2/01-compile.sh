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
# enabled sites, i.e., 4 strains, resembling allele-evoepi-concept's CPU ABM
# (2026-05-20-founder notebook) (genome_alleleevoepi_2026_11_22): infection
# is memoryless, recovering w.p. 0.1 per generation (mean 10 generations);
# immunity to each allele is partial, a level held on a nibble, set to 1 on
# recovery, then decaying by factor 0.99 per generation in expectation
# (probabilistic nibble steps), to a floor of 0.05; each allele's immunity
# multiplies susceptibility by 1 - 0.7 * level; within-host evolution is
# neutral (WITHINHOST_B 0), with WITHINHOST_T 1, so each site of a new
# infection mutates with probability equal to the mutation rate, regardless
# of host immunity (i.e., mutation rate is per site per transmission)
# debug, part 2: as 2026-11-22-alleleevoepi-partialimmunity-debug, whose job 6
# (p2p batches of 24 genomes, 6-wavelet p2p messages) hangs, but with p2p
# batches of 8 genomes (4-wavelet p2p messages), as its job 5, which doesn't;
# jobs bring in the remaining features one at a time, all on the full grid (no
# subgrids), mutation rate 1e-6, infection seeded on corner PE only,
# population 1055 per PE, 10000 updates, 4-bit pathogen nibble messages, p2p
# send probability 1 in 20, each job's features adding to the previous job's:
# 1. control, as debug job 5: 1 genome per neighbor send, recv batches of 4,
#    neighbors sent any genomes (with recv throttling)
# 2. neighbor sends of 8 genomes (1 wavelet), received 2 sends at a time
# 3. neighbors sent only live genomes, each free send thread sending to the
#    least sent-to direction, with sends quelled for empty hosts passed over
#    and no recv throttling
# per-PE trait logger records the 3-bit trait value of each sampled host
# (0th population member), one multiplexed record per bit: bit 0 set if
# infected, and bits 1 and 2 the alleles at sites 0 and 1
# tournament size 1, so each host's update samples one random contact, as in
# the ABM
# note: globalSeed is u16 in kernel, and globalSeed * nRow * nCol is computed
# as u32, so seed must stay below 2**32 / (1170 * 755) ~= 4862 to avoid wrap
genome_flavor=genome_alleleevoepi_2026_11_22
p2pMode=pairs
nrow=1170
ncol=755
p2p=0.05  # compiled-in, runtime config sets it per job
subgrid=0
seedstride=1170  # only corner PE
popsize=1055
jobnum=0  # job order, in config names, as run step runs them in name order

# compile, for each compile config; and write each run config's runtime
# config, with compiled artifacts, to its own directory (one job per run
# config in run step)
for compileConfig in p2pbatch8 neighborbatch live; do
    # message bits, genomes per neighbor send and recv, genomes per p2p
    # message, live neighbor sends?
    case "${compileConfig}" in
        p2pbatch8) msgbits=4 sendgenomes=1 recvgenomes=4 p2pbatch=8 live=0 ;;
        neighborbatch) msgbits=4 sendgenomes=8 recvgenomes=16 p2pbatch=8 live=0 ;;
        live) msgbits=4 sendgenomes=8 recvgenomes=16 p2pbatch=8 live=1 ;;
    esac
    compileseed=$((1150 + jobnum))  # runtime config sets it too
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
export ASYNC_GA_NCYCLE_AT_LEAST=10000
export ASYNC_GA_POPSIZE=${popsize}
export ASYNC_GA_POPULATION_FLAVOR="population_tournamentsteadystate"
export ASYNC_GA_TOURNSIZE_NUMERATOR=1
export ASYNC_GA_TOURNSIZE_DENOMINATOR=1
export ASYNC_GA_GLOBAL_SEED=${compileseed}
export COMPCONFENV_CEREBRASLIB_POPULATION_EXTINCTION_PROBABILITY__f32="0.0"
export COMPCONFENV_CEREBRASLIB_TRAITLOGGER_DSTREAM_ALGO_NAME__comptime_string="hybrid_0_steady_1_stretched_2_algo"
export COMPCONFENV_CEREBRASLIB_TRAITLOGGER_NUM_BITS__u32=1024
export COMPCONFENV_CEREBRASLIB_TRAITLOGGER_DILATION__u32=1  # no dilation: sample every cycle
export COMPCONFENV_CEREBRASLIB_TRAITLOGGER_MULTIPLEX_LSB__u16=3
export COMPCONFENV_CEREBRASLIB_EVOEPI_N_SITES__u16=2
export COMPCONFENV_CEREBRASLIB_EVOEPI_SEED_PE_STRIDE__u16=${seedstride}
export COMPCONFENV_CEREBRASLIB_EVOEPI_CONTACT_RATE__f32="0.35"
export COMPCONFENV_CEREBRASLIB_EVOEPI_IMMUNE_STRENGTH__f32="0.7"
export COMPCONFENV_CEREBRASLIB_EVOEPI_RECOVERY_RATE__f32="0.1"
export COMPCONFENV_CEREBRASLIB_EVOEPI_WANING_RATE__f32="0.01"
export COMPCONFENV_CEREBRASLIB_EVOEPI_IMMUNITY_FLOOR__f32="0.05"
export COMPCONFENV_CEREBRASLIB_EVOEPI_MSG_BITS__u16=${msgbits}
export COMPCONFENV_CEREBRASLIB_EVOEPI_WITHINHOST_B__f32="0.0"
export COMPCONFENV_CEREBRASLIB_EVOEPI_WITHINHOST_T__f32="1.0"
export COMPCONFENV_CEREBRASLIB_EVOEPI_MUTATION_RATE__f32="1e-6"  # runtime config overrides
export COMPCONFENV_ASYNC_GA_P2P_SEND_PROBABILITY__f32="${p2p}"
export COMPCONFENV_ASYNC_GA_NEIGHBOR_SEND_NUM_GENOMES__u16=${sendgenomes}
export COMPCONFENV_ASYNC_GA_NEIGHBOR_RECV_NUM_GENOMES__u16=${recvgenomes}
export COMPCONFENV_ASYNC_GA_NEIGHBOR_SEND_LIVE__u16=${live}
export COMPCONFENV_ASYNC_GA_P2P_BATCH_SIZE__u16=${p2pbatch}

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
echo "COMPCONFENV_CEREBRASLIB_EVOEPI_CONTACT_RATE__f32 \${COMPCONFENV_CEREBRASLIB_EVOEPI_CONTACT_RATE__f32}"
echo "COMPCONFENV_CEREBRASLIB_EVOEPI_IMMUNE_STRENGTH__f32 \${COMPCONFENV_CEREBRASLIB_EVOEPI_IMMUNE_STRENGTH__f32}"
echo "COMPCONFENV_CEREBRASLIB_EVOEPI_RECOVERY_RATE__f32 \${COMPCONFENV_CEREBRASLIB_EVOEPI_RECOVERY_RATE__f32}"
echo "COMPCONFENV_CEREBRASLIB_EVOEPI_WANING_RATE__f32 \${COMPCONFENV_CEREBRASLIB_EVOEPI_WANING_RATE__f32}"
echo "COMPCONFENV_CEREBRASLIB_EVOEPI_IMMUNITY_FLOOR__f32 \${COMPCONFENV_CEREBRASLIB_EVOEPI_IMMUNITY_FLOOR__f32}"
echo "COMPCONFENV_CEREBRASLIB_EVOEPI_MSG_BITS__u16 \${COMPCONFENV_CEREBRASLIB_EVOEPI_MSG_BITS__u16}"
echo "COMPCONFENV_CEREBRASLIB_EVOEPI_WITHINHOST_B__f32 \${COMPCONFENV_CEREBRASLIB_EVOEPI_WITHINHOST_B__f32}"
echo "COMPCONFENV_CEREBRASLIB_EVOEPI_WITHINHOST_T__f32 \${COMPCONFENV_CEREBRASLIB_EVOEPI_WITHINHOST_T__f32}"
echo "COMPCONFENV_CEREBRASLIB_EVOEPI_MUTATION_RATE__f32 \${COMPCONFENV_CEREBRASLIB_EVOEPI_MUTATION_RATE__f32}"
echo "COMPCONFENV_ASYNC_GA_P2P_SEND_PROBABILITY__f32 \${COMPCONFENV_ASYNC_GA_P2P_SEND_PROBABILITY__f32}"
echo "COMPCONFENV_ASYNC_GA_NEIGHBOR_SEND_NUM_GENOMES__u16 \${COMPCONFENV_ASYNC_GA_NEIGHBOR_SEND_NUM_GENOMES__u16}"
echo "COMPCONFENV_ASYNC_GA_NEIGHBOR_RECV_NUM_GENOMES__u16 \${COMPCONFENV_ASYNC_GA_NEIGHBOR_RECV_NUM_GENOMES__u16}"
echo "COMPCONFENV_ASYNC_GA_NEIGHBOR_SEND_LIVE__u16 \${COMPCONFENV_ASYNC_GA_NEIGHBOR_SEND_LIVE__u16}"
echo "COMPCONFENV_ASYNC_GA_P2P_BATCH_SIZE__u16 \${COMPCONFENV_ASYNC_GA_P2P_BATCH_SIZE__u16}"
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
    RUN_CONFIGS="${compileConfig}:0.05:none:$((1150 + jobnum))"
    nest=""
    for runConfig in ${RUN_CONFIGS}; do
        IFS=: read -r runName runP2p runMutexp runSeed <<< "${runConfig}"
        jobnum=$((jobnum + 1))
        CONFIG_NAME="flavor=${genome_flavor}+job=${jobnum}+run=${runName}+msgbits=${msgbits}+sendgenomes=${sendgenomes}+p2pbatch=${p2pbatch}+live=${live}+seed=${runSeed}"
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
        echo "{\"dispatch_slug\": \"${runName}\", \"p2pMode\": \"${p2pMode}\"${nest}, \"p2pSendProbability\": ${runP2p}, \"globalSeed\": ${runSeed}, \"setSymbolValues\": {\"rtc_evoepi_mutationRate\": \"lambda shape: np.full(shape, ${mutation}, np.float32)\"}}" \
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
