#!/bin/bash
# Run a command, retrying it up to ASYNC_GA_KERNEL_STALL_RETRIES (default 4)
# more times, but only if it fails with the Cerebras SDK runtime's kernel
# stall error, e.g.,
#   the received length (0 bytes) is not expected (16 bytes), could be a
#   kernel stall
# Any other failure is returned immediately. For CI use only.
#
# On each stall, prints simulator diagnostics written during that attempt
# (sim_stats.json, e.g., sim_stop_cause and cycle_count, and the tail of
# sim.log), before a retry overwrites them; it also copies them to
# ${ASYNC_GA_KERNEL_STALL_DIAGNOSTICS_DIR:-stall_diagnostics}/attempt=<n>/,
# e.g., for upload as a CI artifact.
#
# usage: retry_on_kernel_stall.sh <command> [args...]

set -uo pipefail

max_retries="${ASYNC_GA_KERNEL_STALL_RETRIES:-4}"
stall_pattern="could be a kernel stall"

log="$(mktemp)"
marker="$(mktemp)"  # attempt start time, to find files the attempt wrote
trap 'rm -f "${log}" "${marker}"' EXIT
diagnostics_dir="${ASYNC_GA_KERNEL_STALL_DIAGNOSTICS_DIR:-stall_diagnostics}"

print_stall_diagnostics() {  # attempt number
    echo "::group::kernel stall diagnostics (attempt ${1})"
    local outdir="${diagnostics_dir}/attempt=${1}"
    mkdir -p "${outdir}"
    cp "${log}" "${outdir}/run.log" || :
    find . -path "./${diagnostics_dir}" -prune -o \( -name sim_stats.json \
        -o -name sim.log -o -name simconfig.json \) -newer "${marker}" \
        -print 2>/dev/null | while read -r path; do
        echo "--- ${path}"
        case "${path}" in
            *.log) tail -n 50 "${path}" ;;
            *) cat "${path}" ;;
        esac
        cp "${path}" "${outdir}/$(echo "${path#./}" | tr / _)" || :
    done
    echo "::endgroup::"
}

status=1
for attempt in $(seq 0 "${max_retries}"); do
    if [ "${attempt}" -gt 0 ]; then
        echo "::warning::kernel stall detected, retrying (retry ${attempt} of ${max_retries}): $*"
    fi
    touch "${marker}"
    "$@" 2>&1 | tee "${log}"
    status="${PIPESTATUS[0]}"
    if [ "${status}" -eq 0 ]; then
        exit 0
    elif ! grep -qF "${stall_pattern}" "${log}"; then
        exit "${status}"  # not a kernel stall, so don't retry
    fi
    print_stall_diagnostics "${attempt}"
done

echo "::error::kernel stall persisted after ${max_retries} retries: $*"
exit "${status}"
