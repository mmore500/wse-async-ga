#!/bin/bash

set -e

cd "$(dirname "$0")"

echo "CS_PYTHON ${CS_PYTHON}"
echo "ASYNC_GA_GENOME_FLAVOR ${ASYNC_GA_GENOME_FLAVOR}"
echo "ASYNC_GA_EXECUTE_FLAGS ${ASYNC_GA_EXECUTE_FLAGS:-}"

export SINGULARITYENV_ASYNC_GA_NCOL="${ASYNC_GA_NCOL:-3}"
export SINGULARITYENV_ASYNC_GA_NROW="${ASYNC_GA_NROW:-3}"
export SINGULARITYENV_ASYNC_GA_NWAV="${ASYNC_GA_NWAV:-3}"
export SINGULARITYENV_ASYNC_GA_NTRAIT="${ASYNC_GA_NTRAIT:-1}"
export SINGULARITY_BINDPATH="../cerebraslib:/cerebraslib"

"${CS_PYTHON}" client.py ${ASYNC_GA_EXECUTE_FLAGS:-}
