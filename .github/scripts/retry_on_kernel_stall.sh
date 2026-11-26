#!/bin/bash
# Run a command, retrying it up to ASYNC_GA_KERNEL_STALL_RETRIES (default 4)
# more times, but only if it fails one of the ways the Cerebras SDK runtime
# intermittently does:
#   the received length (0 bytes) is not expected (16 bytes), could be a
#   kernel stall
# or an abort while copying data back, after an otherwise complete run,
#   terminate called recursively
#   terminate called after throwing an instance of 'std::runtime_error'
# Any other failure is returned immediately. For CI use only.
#
# On each such failure, prints simulator diagnostics written during that attempt
# (sim_stats.json, e.g., sim_stop_cause and cycle_count, and the tail of
# sim.log), before a retry overwrites them; also copy them to
# ${ASYNC_GA_KERNEL_STALL_DIAGNOSTICS_DIR:-stall_diagnostics}/attempt=<n>/,
# e.g., for upload as a CI artifact.
#
# usage: retry_on_kernel_stall.sh <command> [args...]

set -uo pipefail

max_retries="${ASYNC_GA_KERNEL_STALL_RETRIES:-4}"
retry_patterns=("could be a kernel stall" "terminate called")

log="$(mktemp)"
marker="$(mktemp)"  # attempt start time, to find files the attempt wrote
trap 'rm -f "${log}" "${marker}"' EXIT
diagnostics_dir="${ASYNC_GA_KERNEL_STALL_DIAGNOSTICS_DIR:-stall_diagnostics}"

is_retryable() {  # does the attempt's log hold a failure worth retrying?
    local pattern
    for pattern in "${retry_patterns[@]}"; do
        grep -qF "${pattern}" "${log}" && return 0
    done
    return 1
}

print_stall_diagnostics() {  # attempt number
    echo "::group::simulator diagnostics (attempt ${1})"
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
        echo "::warning::simulator failure, retrying (retry ${attempt} of ${max_retries}): $*"
    fi
    touch "${marker}"
    "$@" 2>&1 | tee "${log}"
    status="${PIPESTATUS[0]}"
    if [ "${status}" -eq 0 ]; then
        exit 0
    elif ! is_retryable; then
        exit "${status}"  # not a known flake, so don't retry
    fi
    print_stall_diagnostics "${attempt}"
done

echo "::error::simulator failure persisted after ${max_retries} retries: $*"
exit "${status}"
