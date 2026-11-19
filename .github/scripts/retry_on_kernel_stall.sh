#!/bin/bash
# Run a command, retrying it up to ASYNC_GA_KERNEL_STALL_RETRIES (default 2)
# more times, but only if it fails with the Cerebras SDK runtime's kernel
# stall error, e.g.,
#   the received length (0 bytes) is not expected (16 bytes), could be a
#   kernel stall
# Any other failure is returned immediately. For CI use only.
#
# usage: retry_on_kernel_stall.sh <command> [args...]

set -uo pipefail

max_retries="${ASYNC_GA_KERNEL_STALL_RETRIES:-2}"
stall_pattern="could be a kernel stall"

log="$(mktemp)"
trap 'rm -f "${log}"' EXIT

status=1
for attempt in $(seq 0 "${max_retries}"); do
    if [ "${attempt}" -gt 0 ]; then
        echo "::warning::kernel stall detected, retrying (retry ${attempt} of ${max_retries}): $*"
    fi
    "$@" 2>&1 | tee "${log}"
    status="${PIPESTATUS[0]}"
    if [ "${status}" -eq 0 ]; then
        exit 0
    elif ! grep -qF "${stall_pattern}" "${log}"; then
        exit "${status}"  # not a kernel stall, so don't retry
    fi
done

echo "::error::kernel stall persisted after ${max_retries} retries: $*"
exit "${status}"
