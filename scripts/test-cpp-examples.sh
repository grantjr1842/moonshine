#!/usr/bin/env bash
# Build and run the two C++ examples against the freshly-built
# ``core/build/libmoonshine.so``.
#
# Usage:
#   scripts/test-cpp-examples.sh
#
# Prerequisites:
#   scripts/test-core.sh must have been run at least once so that
#   ``core/build/libmoonshine.so`` exists. We use that build (not
#   a downloaded release tarball) so the example is exercised
#   against the same code as the rest of the test suite.
#
# Environment:
#   CXX                      C++ compiler to use (default: g++).
#   CXXFLAGS                 Extra flags (default: -O2 -std=c++17).
#   SKIP_TTS_EXAMPLE=1       Skip the text-to-speech example.
#   TEST_EXAMPLES_KEEP_GOING=1   Don't abort on first FAIL.
#
# Exit codes:
#   0  all passed
#   1  one or more examples failed
#   77  prerequisites missing (libmoonshine.so not built)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CXX_BIN="${CXX:-g++}"
CXXFLAGS_VALUE="${CXXFLAGS:--O2 -std=c++17}"

log() { echo "[test-cpp-examples] $*" >&2; }
die() { echo "[test-cpp-examples] ERROR: $*" >&2; exit 1; }

cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

LIBMOONSHINE="${REPO_ROOT}/core/build/libmoonshine.so"
LIBORT_DIR_LINUX_X64="${REPO_ROOT}/core/third-party/onnxruntime/lib/linux/x86_64"
LIBORT_DIR_LINUX_ARM64="${REPO_ROOT}/core/third-party/onnxruntime/lib/linux/aarch64"
LIBORT_DIR_MACOS_ARM64="${REPO_ROOT}/core/third-party/onnxruntime/lib/macos-arm64"
LIBORT_DIR_MACOS_X64="${REPO_ROOT}/core/third-party/onnxruntime/lib/macos-x86_64"

if [[ ! -f "${LIBMOONSHINE}" ]]; then
    log "SKIP: ${LIBMOONSHINE} not built"
    log "      run scripts/test-core.sh first to enable C++ example tests"
    exit 77
fi

if ! command -v "${CXX_BIN}" >/dev/null 2>&1; then
    die "C++ compiler not found: ${CXX_BIN}"
fi

if [[ ! -f "${REPO_ROOT}/core/moonshine-cpp.h" ]]; then
    die "missing ${REPO_ROOT}/core/moonshine-cpp.h"
fi

# Pick the ORT lib for the host platform.
case "$(uname -s):$(uname -m)" in
    Linux:x86_64)   LIBORT_DIR="${LIBORT_DIR_LINUX_X64}" ;;
    Linux:aarch64)  LIBORT_DIR="${LIBORT_DIR_LINUX_ARM64}" ;;
    Linux:arm64)    LIBORT_DIR="${LIBORT_DIR_LINUX_ARM64}" ;;
    Darwin:arm64)   LIBORT_DIR="${LIBORT_DIR_MACOS_ARM64}" ;;
    Darwin:x86_64)  LIBORT_DIR="${LIBORT_DIR_MACOS_X64}" ;;
    *)              LIBORT_DIR="" ;;
esac

if [[ -n "${LIBORT_DIR}" && ! -d "${LIBORT_DIR}" ]]; then
    log "warning: onnxruntime lib dir not found at ${LIBORT_DIR}"
    log "         the linker may still find it via the build tree"
fi

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/moonshine-test-cpp-examples.XXXXXX")"
log "workdir: ${WORKDIR}"
trap 'rm -rf "${WORKDIR}"' EXIT

log "compiler: ${CXX_BIN} ${CXXFLAGS_VALUE}"

build_cpp_example() {
    local source="$1"
    local binary_name="$2"
    local binary="${WORKDIR}/${binary_name}"
    log "compiling $source → $binary_name"
    if ! "${CXX_BIN}" ${CXXFLAGS_VALUE} \
        -I"${REPO_ROOT}/core" \
        -L"${REPO_ROOT}/core/build" \
        -Wl,-rpath,"${REPO_ROOT}/core/build" \
        -Wl,-rpath,"${LIBORT_DIR}" \
        "${source}" \
        -lmoonshine \
        -o "${binary}" \
        2>"${WORKDIR}/${binary_name}.build.log"; then
        log "BUILD FAILED for $source — see ${WORKDIR}/${binary_name}.build.log"
        sed 's/^/  /' "${WORKDIR}/${binary_name}.build.log" >&2
        RESULTS_NAMES+=("$binary_name")
        RESULTS_VERDICTS+=("FAIL")
        RESULTS_DETAILS+=("build failed")
        RESULTS_DURATIONS+=("0.0")
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Run + verdict
# ---------------------------------------------------------------------------

declare -a RESULTS_NAMES=()
declare -a RESULTS_VERDICTS=()
declare -a RESULTS_DETAILS=()
declare -a RESULTS_DURATIONS=()

run_cpp_example() {
    local name="$1"
    local binary="$2"
    shift 2
    local start end dur
    local out_file
    out_file="$(mktemp)"

    log "running $name"
    start=$(date +%s.%N)
    local verdict=""
    local detail=""
    if LD_LIBRARY_PATH="${REPO_ROOT}/core/build:${LIBORT_DIR}" \
       timeout 120 "${binary}" "$@" >"${out_file}" 2>&1; then
        # Determine the verdict by inspecting the output. The
        # examples don't have a hard PASS/FAIL convention, so we
        # look for a positive signal in their output:
        #   transcriber: at least one "Line completed:" line
        #   tts: a non-empty output.wav file
        # Both checks below are post-build, so a build-pass
        # combined with a run-pass means PASS.
        case "$name" in
            transcriber)
                if grep -q "Line completed:" "${out_file}"; then
                    verdict="PASS"
                    detail="Line completed observed"
                else
                    verdict="FAIL"
                    detail="no 'Line completed:' line"
                fi
                ;;
            text-to-speech)
                # The example writes to --output; for the smoke
                # test we redirect to a temp file and check size.
                # The caller passes the output path as the last
                # argument; we capture it here.
                local out_path="${!#}"  # last argument
                if [[ -s "${out_path}" ]]; then
                    verdict="PASS"
                    detail="$(wc -c <"${out_path}") bytes"
                else
                    verdict="FAIL"
                    detail="no output.wav at ${out_path}"
                fi
                ;;
            *)
                # Unknown binary: trust the exit code.
                verdict="PASS"
                detail="exit 0"
                ;;
        esac
    else
        local rc=$?
        verdict="FAIL"
        detail="exit $rc"
    fi
    end=$(date +%s.%N)
    dur=$(awk "BEGIN { printf \"%.1f\", ${end} - ${start} }")

    RESULTS_NAMES+=("$name")
    RESULTS_VERDICTS+=("$verdict")
    RESULTS_DETAILS+=("$detail")
    RESULTS_DURATIONS+=("$dur")
    echo "${verdict}: ${name} (cpp, ${detail})"
    rm -f "${out_file}"
}

# ---------------------------------------------------------------------------
# Build + run transcriber
# ---------------------------------------------------------------------------

# The transcriber's AudioProducer expects a 16 kHz WAV; test-assets/two_cities.wav
# is 48 kHz mono. The C++ example reads it via its loadWavData() which
# resamples internally (via the moonshine resampler).
TRANSCRIBER_MODEL_DIR="${REPO_ROOT}/test-assets/tiny-en"
TRANSCRIBER_WAV="${REPO_ROOT}/test-assets/two_cities.wav"

if [[ ! -d "${TRANSCRIBER_MODEL_DIR}" ]]; then
    log "warning: ${TRANSCRIBER_MODEL_DIR} missing — transcriber test will FAIL"
fi
if [[ ! -f "${TRANSCRIBER_WAV}" ]]; then
    die "missing ${TRANSCRIBER_WAV} — pull LFS fixtures first"
fi

# Build into the workdir (the example writes output.wav to cwd).
cd "${WORKDIR}"
if build_cpp_example \
    "${REPO_ROOT}/examples/c++/transcriber.cpp" \
    "transcriber"; then
    run_cpp_example "transcriber" \
        "${WORKDIR}/transcriber" \
        --model-path "${TRANSCRIBER_MODEL_DIR}" \
        --wav-path "${TRANSCRIBER_WAV}"
fi

# ---------------------------------------------------------------------------
# Build + run text-to-speech
# ---------------------------------------------------------------------------

if [[ "${SKIP_TTS_EXAMPLE:-0}" != "1" ]]; then
    TTS_ASSET_ROOT="${REPO_ROOT}/core/moonshine-tts/data"
    TTS_OUT="${WORKDIR}/tts_output.wav"
    if [[ ! -d "${TTS_ASSET_ROOT}" ]]; then
        log "warning: ${TTS_ASSET_ROOT} missing — TTS test will FAIL"
    fi

    if build_cpp_example \
        "${REPO_ROOT}/examples/c++/text-to-speech.cpp" \
        "text-to-speech"; then
        run_cpp_example "text-to-speech" \
            "${WORKDIR}/text-to-speech" \
            --asset-root "${TTS_ASSET_ROOT}" \
            --text "Hello, world." \
            --output "${TTS_OUT}"
    fi
else
    log "SKIP_TTS_EXAMPLE=1 — skipping text-to-speech example"
    RESULTS_NAMES+=("text-to-speech")
    RESULTS_VERDICTS+=("SKIP")
    RESULTS_DETAILS+=("SKIP_TTS_EXAMPLE=1")
    RESULTS_DURATIONS+=("0.0")
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

echo "" >&2
echo "[test-cpp-examples] ===== summary =====" >&2
printf "%-12s  %-32s  %7s\n" "verdict" "example" "secs" >&2
printf "%-12s  %-32s  %7s\n" "-------" "-------" "----" >&2
for i in "${!RESULTS_NAMES[@]}"; do
    printf "%-12s  %-32s  %7s\n" \
        "${RESULTS_VERDICTS[$i]}" \
        "${RESULTS_NAMES[$i]}" \
        "${RESULTS_DURATIONS[$i]}" >&2
    case "${RESULTS_VERDICTS[$i]}" in
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
if [[ "${PASS_COUNT}" -eq 0 && "${SKIP_COUNT}" -gt 0 ]]; then
    log "all examples SKIP (likely missing dependencies)"
    exit 77
fi
log "OK — all ${PASS_COUNT} example(s) passed"
