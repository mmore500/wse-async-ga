#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"  # the wheelhouse dir

# Populates this directory with the wheels pinned in requirements.txt

# SDK 2.10.0 ships CPython 3.11
PYTHON_VERSION="${ASYNC_GA_WHEEL_PYTHON_VERSION:-3.11}"
PLATFORMS="${ASYNC_GA_WHEEL_PLATFORMS:-manylinux2014_x86_64 manylinux_2_24_aarch64}"
echo "PYTHON_VERSION ${PYTHON_VERSION}"
echo "PLATFORMS ${PLATFORMS}"

PIP_DOWNLOAD_ARGS=(
    --dest .
    --no-deps  # do not drag in (and shadow) cs_python's own numpy etc.
    --only-binary=:all:
    --python-version "${PYTHON_VERSION}"
    --implementation cp
    --platform any
    --requirement requirements.txt
)

for platform in ${PLATFORMS}; do
    echo "fetching wheels for ${platform} ..."
    # first try to satisfy from wheels already in this directory (no network)
    if python3 -m pip download --quiet --no-index --find-links . \
        "${PIP_DOWNLOAD_ARGS[@]}" --platform "${platform}" 2>/dev/null; then
        echo "  already present"
    else  # ... otherwise download
        python3 -m pip download \
            "${PIP_DOWNLOAD_ARGS[@]}" --platform "${platform}"
    fi
done

ls -l .
du -sh .
