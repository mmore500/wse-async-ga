#!/bin/bash

set -e

cd "$(dirname "$0")"

if [ -z "$1" ]; then echo "NBEN arg not supplied!"; exit 1; fi
NBEN="$1"
echo "NBEN ${NBEN}"

echo "SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID}"

WSE_SKETCHES_REVISION="fc8ba45ad5ace1412a3daea2afb9be3d999aefb0"
echo "WSE_SKETCHES_REVISION ${WSE_SKETCHES_REVISION}"

WORKDIR="${HOME}/scratch/2024-10-17-cupy-5050-cpu-1k/nben=${NBEN}+async-ga-hypermutator-512"
echo "WORKDIR ${WORKDIR}"

export CSLC="${CSLC:-cslc}"
echo "CSLC ${CSLC}"

echo "initialization telemetry ==============================================="
echo "hostname $(hostname)"
module purge || :
module load Python/3.10.8 || :
echo "python3.10 \$(which python3.10)"
echo "python3.10 --version \$(python3.10 --version)"

echo "setup WORKDIR =========================================================="
mkdir -p "${WORKDIR}"

echo "setup SOURCEDIR ========================================================"
SOURCEDIR="/tmp/${WSE_SKETCHES_REVISION}-${SLURM_JOB_ID}"
echo "SOURCEDIR ${SOURCEDIR}"
rm -rf "${SOURCEDIR}"
git clone https://github.com/mmore500/wse-sketches.git "${SOURCEDIR}" --single-branch -b cupy || git clone https://github.com/mmore500/wse-sketches.git "${SOURCEDIR}" --single-branch
cd "${SOURCEDIR}"
git checkout "${WSE_SKETCHES_REVISION}"
cd -

echo "begin work loop ========================================================"
seed=0
export ASYNC_GA_NCOL=81
echo "ASYNC_GA_NCOL ${ASYNC_GA_NCOL}"

export ASYNC_GA_NROW=81
echo "ASYNC_GA_NROW ${ASYNC_GA_NROW}"

for config in \
    "export ASYNC_GA_NCOL_SUBGRID=1 ASYNC_GA_NROW_SUBGRID=1 NREP=1" \
    "export ASYNC_GA_NCOL_SUBGRID=3 ASYNC_GA_NROW_SUBGRID=3 NREP=1" \
    "export ASYNC_GA_NCOL_SUBGRID=9 ASYNC_GA_NROW_SUBGRID=9 NREP=1" \
    "export ASYNC_GA_NCOL_SUBGRID=27 ASYNC_GA_NROW_SUBGRID=27 NREP=1" \
    "export ASYNC_GA_NCOL_SUBGRID=81 ASYNC_GA_NROW_SUBGRID=81 NREP=1" \
; do
eval "${config}"
echo "ASYNC_GA_NCOL ${ASYNC_GA_NCOL}"
echo "ASYNC_GA_NROW ${ASYNC_GA_NROW}"
echo "ASYNC_GA_NCOL_SUBGRID ${ASYNC_GA_NCOL_SUBGRID}"
echo "ASYNC_GA_NROW_SUBGRID ${ASYNC_GA_NROW_SUBGRID}"
echo "NREP ${NREP}"
for rep in $(seq 1 ${NREP}); do
echo "rep ${rep}"
seed=$((seed+1))
echo "seed ${seed}"

SLUG="wse-sketches+subgrid=${ASYNC_GA_NCOL_SUBGRID}+seed=${seed}"
echo "SLUG ${SLUG}"

echo "configure kernel compile ==============================================="
rm -rf "${WORKDIR}/${SLUG}"
cp -r "${SOURCEDIR}" "${WORKDIR}/${SLUG}"
cd "${WORKDIR}/${SLUG}"
git status

export ASYNC_GA_FABRIC_DIMS="757,996"
echo "ASYNC_GA_FABRIC_DIMS ${ASYNC_GA_FABRIC_DIMS}"

export ASYNC_GA_ARCH_FLAG="--arch=wse2"
echo "ASYNC_GA_ARCH_FLAG ${ASYNC_GA_ARCH_FLAG}"

export ASYNC_GA_GENOME_FLAVOR="genome_cupy_${NBEN}xl_5050_poisson"
echo "ASYNC_GA_GENOME_FLAVOR ${ASYNC_GA_GENOME_FLAVOR}"
export ASYNC_GA_NWAV="${ASYNC_GA_NWAV:-1}"
echo "ASYNC_GA_NWAV ${ASYNC_GA_NWAV}"
export ASYNC_GA_NTRAIT="${ASYNC_GA_NTRAIT:-2}"
echo "ASYNC_GA_NTRAIT ${ASYNC_GA_NTRAIT}"

export ASYNC_GA_MSEC_AT_LEAST=0
echo "ASYNC_GA_MSEC_AT_LEAST ${ASYNC_GA_MSEC_AT_LEAST}"

export ASYNC_GA_NCYCLE_AT_LEAST=1000
echo "ASYNC_GA_NCYCLE_AT_LEAST ${ASYNC_GA_NCYCLE_AT_LEAST}"

export ASYNC_GA_GLOBAL_SEED="${seed}"
echo "ASYNC_GA_GLOBAL_SEED ${ASYNC_GA_GLOBAL_SEED}"

export ASYNC_GA_POPSIZE=256
echo "ASYNC_GA_POPSIZE ${ASYNC_GA_POPSIZE}"

export ASYNC_GA_TOURNSIZE_NUMERATOR=11
echo "ASYNC_GA_TOURNSIZE_NUMERATOR ${ASYNC_GA_TOURNSIZE_NUMERATOR}"

export ASYNC_GA_TOURNSIZE_DENOMINATOR=10
echo "ASYNC_GA_TOURNSIZE_DENOMINATOR ${ASYNC_GA_TOURNSIZE_DENOMINATOR}"

echo "create sbatch file ====================================================="

SBATCH_FILE="$(mktemp)"
echo "SBATCH_FILE ${SBATCH_FILE}"

###############################################################################
# ----------------------------------------------------------------------------#
###############################################################################
cat > "${SBATCH_FILE}" << EOF
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --constraint=amr
#SBATCH --output="/mnt/home/%u/joblog/id=%j+ext=.txt"
#SBATCH --mail-user=mawni4ah2o@pomail.net
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --account=beacon

# nodelist, per https://docs.icer.msu.edu/Cluster_Resources/
# AMD EPYC 7H12 Processor (2.595 GHz) 128 cores, 493 GB memory, 412 GB disck

set -e

echo "cc SLURM script --------------------------------------------------------"
JOBSCRIPT="\${HOME}/jobscript/id=\${SLURM_JOB_ID}+ext=.sbatch"
echo "JOBSCRIPT \${JOBSCRIPT}"
cp "\${0}" "\${JOBSCRIPT}"
chmod +x "\${JOBSCRIPT}"

echo "job telemetry ----------------------------------------------------------"
echo "source SLURM_JOB_ID ${SLURM_JOB_ID}"
echo "current SLURM_JOB_ID \${SLURM_JOB_ID}"
echo "hostname \$(hostname)"
echo "date \$(date)"

echo "lscpu ----------------------------------------------------------"
lscpu || :

echo "lshw ----------------------------------------------------------"
lshw || :

echo "cpuinfo ----------------------------------------------------------"
cat /proc/cpuinfo || :

echo "module setup -----------------------------------------------------------"
module purge || :
module load Python/3.10.8 || :
echo "python3.10 \$(which python3.10)"
echo "python3.10 --version \$(python3.10 --version)"

echo "initialization telemetry -----------------------------------------------"
echo "WSE_SKETCHES_REVISION ${WSE_SKETCHES_REVISION}"
echo "HEAD_REVISION $(git rev-parse HEAD)"
echo "WORKDIR ${WORKDIR}"
echo "SLUG ${SLUG}"
echo "SDK_INSTALL_PATH \${SDK_INSTALL_PATH:-}"

echo "ASYNC_GA_NCOL_SUBGRID ${ASYNC_GA_NCOL_SUBGRID}"
echo "ASYNC_GA_NROW_SUBGRID ${ASYNC_GA_NROW_SUBGRID}"
echo "ASYNC_GA_MSEC_AT_LEAST ${ASYNC_GA_MSEC_AT_LEAST}"
echo "ASYNC_GA_NCYCLE_AT_LEAST ${ASYNC_GA_NCYCLE_AT_LEAST}"
echo "ASYNC_GA_GLOBAL_SEED ${ASYNC_GA_GLOBAL_SEED}"
echo "NREP ${NREP}"
echo "NBEN ${NBEN}"
echo "WSE_SKETCHES_REVISION ${WSE_SKETCHES_REVISION}"

export ASYNC_GA_GENOME_FLAVOR="${ASYNC_GA_GENOME_FLAVOR}"
echo "ASYNC_GA_GENOME_FLAVOR \${ASYNC_GA_GENOME_FLAVOR}"
export ASYNC_GA_NWAV="${ASYNC_GA_NWAV}"
echo "ASYNC_GA_NWAV \${ASYNC_GA_NWAV}"
export ASYNC_GA_NTRAIT="${ASYNC_GA_NTRAIT}"
echo "ASYNC_GA_NTRAIT \${ASYNC_GA_NTRAIT}"
export ASYNC_GA_EXECUTE_FLAGS="--cmaddr \${CS_IP_ADDR}:9000 --no-suptrace"
echo "ASYNC_GA_EXECUTE_FLAGS \${ASYNC_GA_EXECUTE_FLAGS}"

echo "setup WORKDIR ----------------------------------------------------------"
cd "${WORKDIR}/${SLUG}"
echo "PWD \${PWD}"

echo "install dependencies ----------------------------------------------------"
python3.10 -m venv ./env
source ./env/bin/activate
echo "python3.10 \$(which python)"
echo "python3.10 --version \$(python3.10 --version)"

python3.10 -m pip install --upgrade pip setuptools wheel
python3.10 -m pip install --upgrade uv
python3.10 -m uv pip install \
    'more_itertools==10.*' \
    'numpy==1.*' \
    'pandas==1.*' \
    'polars==1.6.*' \
    'pyarrow==15.*' \
    'scipy==1.*' \
    'tqdm==4.*'

echo "pip freeze -------------------------------------------------------------"
python3.10 -m pip freeze

echo "execute kernel program -------------------------------------------------"
export ASYNC_GA_NCOL=${ASYNC_GA_NCOL}
export ASYNC_GA_NROW=${ASYNC_GA_NROW}
export ASYNC_GA_NCOL_SUBGRID=${ASYNC_GA_NCOL_SUBGRID}
export ASYNC_GA_NROW_SUBGRID=${ASYNC_GA_NROW_SUBGRID}
export ASYNC_GA_MSEC_AT_LEAST=${ASYNC_GA_MSEC_AT_LEAST}
export ASYNC_GA_TSC_AT_LEAST=0
export ASYNC_GA_NCYCLE_AT_LEAST=${ASYNC_GA_NCYCLE_AT_LEAST}
export ASYNC_GA_GENOME_FLAVOR=${ASYNC_GA_GENOME_FLAVOR}
export ASYNC_GA_GLOBAL_SEED=${ASYNC_GA_GLOBAL_SEED}
export ASYNC_GA_POPSIZE=${ASYNC_GA_POPSIZE}
export ASYNC_GA_TOURNSIZE_NUMERATOR=${ASYNC_GA_TOURNSIZE_NUMERATOR}
export ASYNC_GA_TOURNSIZE_DENOMINATOR=${ASYNC_GA_TOURNSIZE_DENOMINATOR}
export NBEN=${NBEN}

echo "ASYNC_GA_NCOL \${ASYNC_GA_NCOL}"
echo "ASYNC_GA_NROW \${ASYNC_GA_NROW}"
echo "ASYNC_GA_NCOL_SUBGRID \${ASYNC_GA_NCOL_SUBGRID}"
echo "ASYNC_GA_NROW_SUBGRID \${ASYNC_GA_NROW_SUBGRID}"
echo "ASYNC_GA_MSEC_AT_LEAST \${ASYNC_GA_MSEC_AT_LEAST}"
echo "ASYNC_GA_TSC_AT_LEAST \${ASYNC_GA_TSC_AT_LEAST}"
echo "ASYNC_GA_NCYCLE_AT_LEAST \${ASYNC_GA_NCYCLE_AT_LEAST}"
echo "ASYNC_GA_GENOME_FLAVOR \${ASYNC_GA_GENOME_FLAVOR}"
echo "ASYNC_GA_GLOBAL_SEED \${ASYNC_GA_GLOBAL_SEED}"
echo "ASYNC_GA_POPSIZE \${ASYNC_GA_POPSIZE}"
echo "ASYNC_GA_TOURNSIZE_NUMERATOR \${ASYNC_GA_TOURNSIZE_NUMERATOR}"
echo "ASYNC_GA_TOURNSIZE_DENOMINATOR \${ASYNC_GA_TOURNSIZE_DENOMINATOR}"
echo "NBEN \${NBEN}"

./pyscript/hypermutator-5050.py

echo "cleanup ----------------------------------------------------------------"
rm -rf ./env

echo "finalization telemetry -------------------------------------------------"
echo "SECONDS \${SECONDS}"
echo ">>>fin<<<"

EOF
###############################################################################
# ----------------------------------------------------------------------------#
###############################################################################


echo "submit sbatch file ==============================="
sbatch "${SBATCH_FILE}"

echo "end work loop =========================================================="
done
done

echo "wait ==================================================================="
wait

echo "finalization telemetry ================================================="
echo "SECONDS ${SECONDS}"
echo ">>>fin<<<"
