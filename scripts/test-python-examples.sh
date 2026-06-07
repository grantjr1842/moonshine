#!/usr/bin/env bash
# Verify every Python example in examples/python/ runs end-to-end under
# ``--self-check`` and produces a PASS/FAIL/SKIP verdict on its last line
# of stdout. Each example uses a fake sounddevice shim (no real audio
# hardware required) and canned test audio from test-assets/.
#
# Usage:
#   scripts/test-python-examples.sh
#
# Environment:
#   MOONSHINE_SELF_CHECK=1       Set automatically before each invocation.
#   PYTHONPATH                    Set automatically to include examples/python.
#   SKIP_POCKET_TTS=1            Skip the pocket_tts_verify unit tests.
#   SKIP_INDIVIDUAL_EXAMPLES=... Comma-separated dotted module names to skip.
#   TEST_EXAMPLES_KEEP_GOING=1   Don't abort on first FAIL.
#
# Exit codes:
#   0  all passed
#   1  one or more examples failed
#   77  all examples were SKIP (e.g. dependencies missing)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHONPATH_VALUE="${REPO_ROOT}/examples/python"
PYTHON_BIN="${PYTHON_BIN:-python3}"

log() { echo "[test-python-examples] $*" >&2; }
die() { echo "[test-python-examples] ERROR: $*" >&2; exit 1; }

cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# The list of examples to test. Hard-coded rather than auto-globbed:
# adding a new example should be a deliberate decision (it needs the
# --self-check flag and a working canned-audio path).
# ---------------------------------------------------------------------------

EXAMPLES=(
    "examples.python.basic_transcription"
    "examples.python.mic_transcription"
    "examples.python.intent_recognition"
    "examples.python.dialog_flow"
    "examples.python.stt.01_offline_transcribe"
    "examples.python.stt.02_streaming_transcribe"
    "examples.python.stt.03_live_microphone"
    "examples.python.stt.04_options_and_tuning"
    "examples.python.stt.05_word_timestamps"
    "examples.python.stt.06_multi_stream"
    "examples.python.stt.07_spelling_mode"
    "examples.python.stt.08_intent_recognizer"
    "examples.python.stt.09_dialog_flow"
    "examples.python.stt.10_full_voice_agent"
    "examples.python.stt.11_schema_extraction"
)

# Examples to skip on this invocation. Pulled from
# SKIP_INDIVIDUAL_EXAMPLES= (comma-separated dotted module names).
declare -A SKIP_THIS=()
if [[ -n "${SKIP_INDIVIDUAL_EXAMPLES:-}" ]]; then
    IFS=',' read -ra SKIPS <<< "${SKIP_INDIVIDUAL_EXAMPLES}"
    for s in "${SKIPS[@]}"; do
        SKIP_THIS["$s"]=1
    done
fi

# ---------------------------------------------------------------------------
# Output capture
# ---------------------------------------------------------------------------

declare -a RESULTS_NAMES=()
declare -a RESULTS_VERDICTS=()   # PASS, FAIL, SKIP
declare -a RESULTS_DETAILS=()    # the trailing parenthetical, or empty
declare -a RESULTS_DURATIONS=()  # seconds, as a float

run_example() {
    local mod="$1"
    local start end dur
    local out_file
    out_file="$(mktemp)"

    log "running $mod --self-check"
    start=$(date +%s.%N)
    # The ``-u`` flag forces unbuffered stdout so the driver
    # sees the final PASS/FAIL/SKIP line promptly. The
    # ``MOONSHINE_SELF_CHECK_VERBOSE=1`` env var lets the
    # example's common.run_self_check wrapper print tracebacks
    # to stderr on FAIL — useful when diagnosing a real bug
    # in an example.
    PYTHONPATH="${PYTHONPATH_VALUE}" \
    MOONSHINE_SELF_CHECK=1 \
        timeout 180 "${PYTHON_BIN}" -u -m test_support.run_example \
        "$mod" --self-check \
        >"$out_file" 2>&1 || true
    end=$(date +%s.%N)
    dur=$(awk "BEGIN { printf \"%.1f\", ${end} - ${start} }")

    # The driver parses the *last* non-empty line of stdout.
    # The example's contract is: diagnostic output on stderr,
    # final PASS/FAIL/SKIP line on stdout. The ``-u`` flag
    # plus ``tail -1`` on the merged stream below handles the
    # case where the example prints to stdout before exiting.
    local verdict
    local detail
    local final_line
    final_line="$(grep -E '^(PASS|FAIL|SKIP):' "$out_file" | tail -1)"
    if [[ -z "$final_line" ]]; then
        # Fall back to the very last non-empty line of the
        # merged output. This catches the case where the
        # example failed before printing a verdict (e.g.
        # import error or unhandled exception).
        final_line="$(grep -v '^$' "$out_file" | tail -1)"
        if [[ -z "$final_line" ]]; then
            final_line="(no output)"
        fi
        verdict="FAIL"
        detail="no PASS/FAIL/SKIP line; final line: ${final_line:0:80}"
    else
        verdict="${final_line%%:*}"
        detail="${final_line#*: }"
    fi

    RESULTS_NAMES+=("$mod")
    RESULTS_VERDICTS+=("$verdict")
    RESULTS_DETAILS+=("$detail")
    RESULTS_DURATIONS+=("$dur")
    rm -f "$out_file"

    # Echo the verdict line so users running this interactively
    # can see progress.
    echo "$final_line"
}

run_pocket_tts_metrics() {
    local start end dur
    local out_file
    out_file="$(mktemp)"

    log "running examples.python.pocket_tts_verify.test_metrics (unittest)"
    start=$(date +%s.%N)
    PYTHONPATH="${PYTHONPATH_VALUE}" \
        timeout 60 "${PYTHON_BIN}" -m unittest \
        examples.python.pocket_tts_verify.test_metrics \
        >"$out_file" 2>&1 || true
    end=$(date +%s.%N)
    dur=$(awk "BEGIN { printf \"%.1f\", ${end} - ${start} }")

    local final_line
    final_line="$(tail -1 "$out_file")"
    if grep -qE '^OK$' "$out_file"; then
        RESULTS_NAMES+=("examples.python.pocket_tts_verify.test_metrics")
        RESULTS_VERDICTS+=("PASS")
        RESULTS_DETAILS+=("unittest suite")
        RESULTS_DURATIONS+=("$dur")
        echo "PASS: examples.python.pocket_tts_verify.test_metrics (unittest suite)"
    else
        RESULTS_NAMES+=("examples.python.pocket_tts_verify.test_metrics")
        RESULTS_VERDICTS+=("FAIL")
        RESULTS_DETAILS+=("unittest: ${final_line:0:80}")
        RESULTS_DURATIONS+=("$dur")
        echo "FAIL: examples.python.pocket_tts_verify.test_metrics (${final_line:0:80})"
    fi
    rm -f "$out_file"
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

log "PYTHONPATH=${PYTHONPATH_VALUE}"
log "PYTHON_BIN=${PYTHON_BIN}"
log "test-assets dir: ${REPO_ROOT}/test-assets"

if [[ ! -f "${REPO_ROOT}/test-assets/two_cities.wav" ]]; then
    die "missing test-assets/two_cities.wav — pull LFS fixtures first"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    die "python interpreter not found: ${PYTHON_BIN}"
fi

if ! "${PYTHON_BIN}" -c "import moonshine_voice" 2>/dev/null; then
    die "moonshine_voice not importable; install with: pip install -e python/"
fi

# The driver does NOT use ``set -e`` style aborts between
# examples — we want to see every verdict before deciding to
# fail. But individual example runs ARE wrapped in timeout +
# error capture so a single hang doesn't block the rest.
for mod in "${EXAMPLES[@]}"; do
    if [[ -n "${SKIP_THIS[$mod]:-}" ]]; then
        log "skipping $mod (SKIP_INDIVIDUAL_EXAMPLES)"
        RESULTS_NAMES+=("$mod")
        RESULTS_VERDICTS+=("SKIP")
        RESULTS_DETAILS+=("skipped by SKIP_INDIVIDUAL_EXAMPLES")
        RESULTS_DURATIONS+=("0.0")
        echo "SKIP: $mod (skipped by SKIP_INDIVIDUAL_EXAMPLES)"
        continue
    fi
    run_example "$mod"
done

if [[ "${SKIP_POCKET_TTS:-0}" != "1" ]]; then
    run_pocket_tts_metrics
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

echo "" >&2
echo "[test-python-examples] ===== summary =====" >&2
printf "%-12s  %-58s  %7s\n" "verdict" "example" "secs" >&2
printf "%-12s  %-58s  %7s\n" "-------" "-------" "----" >&2
for i in "${!RESULTS_NAMES[@]}"; do
    printf "%-12s  %-58s  %7s\n" \
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
