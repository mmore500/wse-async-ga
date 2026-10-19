#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

echo "begin compile.sh --------------------------------------------------------"
echo "which python3 $(which python3)"

echo "CSLC ${CSLC}"

ASYNC_GA_GENOME_FLAVOR="${ASYNC_GA_GENOME_FLAVOR:-genome_bitdrift}"
echo "ASYNC_GA_GENOME_FLAVOR ${ASYNC_GA_GENOME_FLAVOR}"

ASYNC_GA_POPULATION_FLAVOR="${ASYNC_GA_POPULATION_FLAVOR:-population_tournamentgenerational}"
echo "ASYNC_GA_POPULATION_FLAVOR ${ASYNC_GA_POPULATION_FLAVOR}"

ASYNC_GA_GLOBAL_SEED="${ASYNC_GA_GLOBAL_SEED:-0}"
echo "ASYNC_GA_GLOBAL_SEED ${ASYNC_GA_GLOBAL_SEED}"

ASYNC_GA_NCYCLE_AT_LEAST="${ASYNC_GA_NCYCLE_AT_LEAST:-40}"
echo "ASYNC_GA_NCYCLE_AT_LEAST ${ASYNC_GA_NCYCLE_AT_LEAST}"

ASYNC_GA_MSEC_AT_LEAST="${ASYNC_GA_MSEC_AT_LEAST:-0}"
echo "ASYNC_GA_MSEC_AT_LEAST ${ASYNC_GA_MSEC_AT_LEAST}"

ASYNC_GA_TSC_AT_LEAST="${ASYNC_GA_TSC_AT_LEAST:-0}"
echo "ASYNC_GA_TSC_AT_LEAST ${ASYNC_GA_TSC_AT_LEAST}"

ASYNC_GA_FABRIC_DIMS="${ASYNC_GA_FABRIC_DIMS:-10,5}"
echo "ASYNC_GA_FABRIC_DIMS ${ASYNC_GA_FABRIC_DIMS}"

ASYNC_GA_NROW="${ASYNC_GA_NROW:-3}"
echo "ASYNC_GA_NROW ${ASYNC_GA_NROW}"

ASYNC_GA_NCOL="${ASYNC_GA_NCOL:-3}"
echo "ASYNC_GA_NCOL ${ASYNC_GA_NCOL}"

ASYNC_GA_NROW_SUBGRID="${ASYNC_GA_NROW_SUBGRID:-0}"
echo "ASYNC_GA_NROW_SUBGRID ${ASYNC_GA_NROW_SUBGRID}"

ASYNC_GA_NCOL_SUBGRID="${ASYNC_GA_NCOL_SUBGRID:-0}"
echo "ASYNC_GA_NCOL_SUBGRID ${ASYNC_GA_NCOL_SUBGRID}"

ASYNC_GA_NONBLOCK="${ASYNC_GA_NONBLOCK:-0}"
echo "ASYNC_GA_NONBLOCK ${ASYNC_GA_NONBLOCK}"

ASYNC_GA_POPSIZE="${ASYNC_GA_POPSIZE:-32}"
echo "ASYNC_GA_POPSIZE ${ASYNC_GA_POPSIZE}"

ASYNC_GA_TOURNSIZE_NUMERATOR="${ASYNC_GA_TOURNSIZE_NUMERATOR:-5}"
echo "ASYNC_GA_TOURNSIZE_NUMERATOR ${ASYNC_GA_TOURNSIZE_NUMERATOR}"

ASYNC_GA_TOURNSIZE_DENOMINATOR="${ASYNC_GA_TOURNSIZE_DENOMINATOR:-1}"
echo "ASYNC_GA_TOURNSIZE_DENOMINATOR ${ASYNC_GA_TOURNSIZE_DENOMINATOR}"

ASYNC_GA_ARCH_FLAG="${ASYNC_GA_ARCH_FLAG:-wse3}"
echo "ASYNC_GA_ARCH_FLAG ${ASYNC_GA_ARCH_FLAG}"
if [[ "${ASYNC_GA_ARCH_FLAG}" == *wse2* ]]; then
    echo "WARNING: wse2 arch unsupported" >&2
fi

ASYNC_GA_CHANNELS_FLAG="${ASYNC_GA_CHANNELS_FLAG:-1}"
echo "ASYNC_GA_CHANNELS_FLAG ${ASYNC_GA_CHANNELS_FLAG}"

ASYNC_GA_FABRIC_OFFSETS_FLAG="${ASYNC_GA_FABRIC_OFFSETS_FLAG:-4,1}"
echo "ASYNC_GA_FABRIC_OFFSETS_FLAG ${ASYNC_GA_FABRIC_OFFSETS_FLAG}"

export COMPCONFENV_CEREBRASLIB_TRAITLOGGER_NUM_BITS__u32="${COMPCONFENV_CEREBRASLIB_TRAITLOGGER_NUM_BITS__u32:-256}"
echo "COMPCONFENV_CEREBRASLIB_TRAITLOGGER_NUM_BITS__u32 ${COMPCONFENV_CEREBRASLIB_TRAITLOGGER_NUM_BITS__u32}"

export COMPCONFENV_CEREBRASLIB_TRAITLOGGER_DILATION__u32="${COMPCONFENV_CEREBRASLIB_TRAITLOGGER_DILATION__u32:-8}"
echo "COMPCONFENV_CEREBRASLIB_TRAITLOGGER_DILATION__u32 ${COMPCONFENV_CEREBRASLIB_TRAITLOGGER_DILATION__u32}"

export COMPCONFENV_CEREBRASLIB_TRAITLOGGER_DSTREAM_ALGO_NAME__comptime_string="${COMPCONFENV_CEREBRASLIB_TRAITLOGGER_DSTREAM_ALGO_NAME__comptime_string:-steady_algo}"
echo "COMPCONFENV_CEREBRASLIB_TRAITLOGGER_DSTREAM_ALGO_NAME__comptime_string ${COMPCONFENV_CEREBRASLIB_TRAITLOGGER_DSTREAM_ALGO_NAME__comptime_string}"

# 0: fossil is 0th population member at collection time
# 1: reservoir sample one genome per generation between fossil collections
export COMPCONFENV_ASYNC_GA_FOSSIL_RESERVOIR__u32="${COMPCONFENV_ASYNC_GA_FOSSIL_RESERVOIR__u32:-0}"
echo "COMPCONFENV_ASYNC_GA_FOSSIL_RESERVOIR__u32 ${COMPCONFENV_ASYNC_GA_FOSSIL_RESERVOIR__u32}"

echo "setup dependency wheelhouse ---------------------------------------------"
CS_PYTHON="${CS_PYTHON:-cs_python}"
echo "CS_PYTHON ${CS_PYTHON}"
if CS_PYTHON_VERSION="$("${CS_PYTHON}" -c \
    'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null \
    | tail -n 1)" && [[ "${CS_PYTHON_VERSION}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    export ASYNC_GA_WHEEL_PYTHON_VERSION="${CS_PYTHON_VERSION}"
else
    echo "could not determine python version from ${CS_PYTHON}, using default"
fi
echo "ASYNC_GA_WHEEL_PYTHON_VERSION ${ASYNC_GA_WHEEL_PYTHON_VERSION:-<default>}"
./wheelhouse/fetch_wheels.sh


pushd ..
git submodule update --init --recursive || :
popd

# symlinks don't work and --import-path doesn't work, so this is a workaround
trap "git checkout ./cerebraslib" EXIT
rsync -rI "$(readlink -f cerebraslib)" .

# target a 2x2 region of interest
# Every program using memcpy must use a fabric offset of 4,1, and if compiling
# for a simulated fabric, must use a fabric dimension of at least
# width+7,height+1, where width and height are the dimensions of the program.
# These additional PEs are used by memcpy to route data on and off the wafer.
# see https://sdk.cerebras.net/csl/tutorials/gemv-01-complete-program/
# 9x4 because compiler says
# RuntimeError: Fabric dimension must be at least 9-by-4

python3 -m compconf --compconf-jq ". += {\"ASYNC_GA_GENOME_FLAVOR:comptime_string\": \"${ASYNC_GA_GENOME_FLAVOR}\", \"ASYNC_GA_POPULATION_FLAVOR:comptime_string\": \"${ASYNC_GA_POPULATION_FLAVOR}\", \"ASYNC_GA_POPSIZE:u32\": ${ASYNC_GA_POPSIZE}}" --compconf-verbose --compconf-cslc "${CSLC}" layout.csl --arch=${ASYNC_GA_ARCH_FLAG} --import-path ./downstream/include --fabric-dims=${ASYNC_GA_FABRIC_DIMS} --fabric-offsets=${ASYNC_GA_FABRIC_OFFSETS_FLAG} --channels=${ASYNC_GA_CHANNELS_FLAG} --memcpy --params=globalSeed:${ASYNC_GA_GLOBAL_SEED},nCycleAtLeast:${ASYNC_GA_NCYCLE_AT_LEAST},msecAtLeast:${ASYNC_GA_MSEC_AT_LEAST},tscAtLeast:${ASYNC_GA_TSC_AT_LEAST},nRow:${ASYNC_GA_NROW},nCol:${ASYNC_GA_NCOL},nRowSubgrid:${ASYNC_GA_NROW_SUBGRID},nColSubgrid:${ASYNC_GA_NCOL_SUBGRID},nonBlock:${ASYNC_GA_NONBLOCK},popSize:${ASYNC_GA_POPSIZE},tournSizeNumerator:${ASYNC_GA_TOURNSIZE_NUMERATOR},tournSizeDenominator:${ASYNC_GA_TOURNSIZE_DENOMINATOR} -o out
