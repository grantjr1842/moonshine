#!/usr/bin/env bash
# Run every example test suite the repo ships, in order:
#   1. C++ core test suite           (scripts/test-core.sh)
#   2. Python example self-checks    (scripts/test-python-examples.sh)
#   3. C++ example build + run       (scripts/test-cpp-examples.sh)
#   4. iOS / Android example builds  (scripts/test-examples.sh)
#
# The umbrella never aborts on individual failure — every sub-suite
# runs to completion, and the aggregated counts are printed at the
# end. Exit code is 0 only if every sub-suite passed.
#
# Usage:
#   scripts/test-all-examples.sh
#
# Environment:
#   SKIP_CORE=1              Skip scripts/test-core.sh
#   SKIP_PYTHON=1            Skip scripts/test-python-examples.sh
#   SKIP_CPP_EXAMPLES=1      Skip scripts/test-cpp-examples.sh
#   SKIP_IOS_ANDROID=1       Skip scripts/test-examples.sh
#                            (or set SKIP_IOS=1 / SKIP_ANDROID=1 to
#                            skip just one platform)
#
# Exit codes:
#   0  all sub-suites passed (or were SKIP)
#   1  one or more sub-suites failed

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

log() { echo "[test-all-examples] $*" >&2; }

declare -a SUITE_NAMES=()
declare -a SUITE_VERDICTS=()
declare -a SUITE_DETAILS=()
declare -a SUITE_DURATIONS=()

run_suite() {
    local name="$1"
    shift
    local start end dur rc
    start=$(date +%s.%N)
    if "$@"; then
        rc=0
    else
        rc=$?
    fi
    end=$(date +%s.%N)
    dur=$(awk "BEGIN { printf \"%.1f\", ${end} - ${start} }")

    case "$rc" in
        0)
            SUITE_VERDICTS+=("PASS")
            SUITE_DETAILS+=("ok")
            log "  $name: PASS (${dur}s)"
            ;;
        77)
            SUITE_VERDICTS+=("SKIP")
            SUITE_DETAILS+=("dependencies missing")
            log "  $name: SKIP (${dur}s)"
            ;;
        *)
            SUITE_VERDICTS+=("FAIL")
            SUITE_DETAILS+=("exit $rc")
            log "  $name: FAIL (exit $rc, ${dur}s)"
            ;;
    esac
    SUITE_NAMES+=("$name")
    SUITE_DURATIONS+=("$dur")
}

log "==== Moonshine example test suite ===="
log "repo root: ${REPO_ROOT}"

cd "${REPO_ROOT}"

# 1. C++ core test suite.
if [[ "${SKIP_CORE:-0}" != "1" ]]; then
    log "[1/4] running scripts/test-core.sh"
    run_suite "test-core.sh" "${REPO_ROOT}/scripts/test-core.sh"
fi

# 2. Python example self-checks.
if [[ "${SKIP_PYTHON:-0}" != "1" ]]; then
    log "[2/4] running scripts/test-python-examples.sh"
    run_suite "test-python-examples.sh" \
        "${REPO_ROOT}/scripts/test-python-examples.sh"
fi

# 3. C++ example build + run.
if [[ "${SKIP_CPP_EXAMPLES:-0}" != "1" ]]; then
    log "[3/4] running scripts/test-cpp-examples.sh"
    run_suite "test-cpp-examples.sh" \
        "${REPO_ROOT}/scripts/test-cpp-examples.sh"
fi

# 4. iOS + Android example builds.
if [[ "${SKIP_IOS_ANDROID:-0}" != "1" ]]; then
    log "[4/4] running scripts/test-examples.sh --local-examples"
    run_suite "test-examples.sh" \
        "${REPO_ROOT}/scripts/test-examples.sh" --local-examples
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

echo "" >&2
echo "[test-all-examples] ===== summary =====" >&2
printf "%-12s  %-32s  %7s\n" "verdict" "suite" "secs" >&2
printf "%-12s  %-32s  %7s\n" "-------" "-----" "----" >&2
for i in "${!SUITE_NAMES[@]}"; do
    printf "%-12s  %-32s  %7s\n" \
        "${SUITE_VERDICTS[$i]}" \
        "${SUITE_NAMES[$i]}" \
        "${SUITE_DURATIONS[$i]}" >&2
    case "${SUITE_VERDICTS[$i]}" in
        PASS) PASS_COUNT=$((PASS_COUNT + 1)) ;;
        FAIL) FAIL_COUNT=$((FAIL_COUNT + 1)) ;;
        SKIP) SKIP_COUNT=$((SKIP_COUNT + 1)) ;;
    esac
done

log "PASS=${PASS_COUNT}  FAIL=${FAIL_COUNT}  SKIP=${SKIP_COUNT}"

if [[ "${FAIL_COUNT}" -gt 0 ]]; then
    log "FAILED — see lines marked FAIL above"
    exit 1
fi
log "OK — all sub-suites passed"
