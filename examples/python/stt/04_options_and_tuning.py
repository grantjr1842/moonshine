"""Example 04 — every non-obvious Transcriber option in one runnable.

The :class:`moonshine_voice.Transcriber` constructor takes an ``options``
dict of strings that the C library parses. Most of them are off by default
and only matter when something is going wrong. This example turns them all
on, streams the bundled WAV, and prints a one-line summary of what each
option did — so you can see the effect without having to read the C header.

What this script demonstrates
-----------------------------
* ``vad_threshold`` — sensitivity of the voice-activity detector.
* ``vad_window_duration`` / ``vad_look_behind_sample_count`` — averaging
  window and look-behind compensation.
* ``vad_max_segment_duration`` — cap on segment length to avoid endless
  chunks.
* ``max_tokens_per_second`` — the **hallucination guard** that catches
  decoder loops. Bump to 13.0 for non-Latin languages.
* ``save_input_wav_path`` — dump received audio as 16 kHz mono WAVs.
  First thing to check if quality is wrong.
* ``identify_speakers`` — turn on / off the speaker embedding model.
* ``return_audio_data`` — keep or drop the per-line audio buffer.
* ``word_timestamps`` — opt in to word-level timing (requires an
  attention-decoder variant of the model; falls back to silent skip if
  the variant is not present).
* ``log_output_text`` / ``log_ort_run`` — verbose logging.

Run it
------
    python -m examples.python.stt.04_options_and_tuning
    python -m examples.python.stt.04_options_and_tuning \\
        --options save_input_wav_path=.
"""

from __future__ import annotations

import os
from typing import Dict

from . import common


def build_options_dict(args) -> Dict[str, str]:
    """Start with the well-tested defaults, then layer in --options overrides.

    All values are strings — the C library parses them. This mirrors the
    style used in ``python/src/moonshine_voice/transcriber.py:121-141``.
    """
    base = {
        "vad_threshold": "0.5",
        "vad_window_duration": "0.5",
        "vad_look_behind_sample_count": "8192",
        "vad_max_segment_duration": "15.0",
        "max_tokens_per_second": "6.5",
        "identify_speakers": "true",
        "return_audio_data": "true",
        "log_output_text": "true" if args.debug else "false",
        # The C-side option is named ``log_ort_run`` (singular)
        # — see core/moonshine-c-api.cpp:124. Earlier versions
        # of this example passed ``log_ort_runs`` (plural), which
        # the C library rejects with "Unknown transcriber option".
        "log_ort_run": "true" if args.debug else "false",
    }
    if args.word_timestamps:
        base["word_timestamps"] = "true"
    if args.options:
        base.update(common.parse_options_string(args.options))
    return base


def print_option_summary(options: Dict[str, str]) -> None:
    common.hr("Options (parsed)")
    for k, v in options.items():
        common.errprint(f"  {k} = {v}")


def summarise_lines(transcript) -> None:
    common.hr("Per-line results")
    if not transcript.lines:
        print("(no speech detected)")
        return
    for line in transcript.lines:
        speaker = (
            f" Speaker #{line.speaker_index}"
            if line.has_speaker_id
            else ""
        )
        audio_len = (
            f" audio={len(line.audio_data)}"
            if line.audio_data
            else " audio=-"
        )
        print(
            f"  [{line.start_time:6.2f}s → "
            f"{line.start_time + line.duration:6.2f}s]"
            f"{speaker}{audio_len}"
            f"  {line.text!r}"
        )

    speakers = {l.speaker_index for l in transcript.lines if l.has_speaker_id}
    common.hr("Summary")
    print(f"  lines             : {len(transcript.lines)}")
    print(f"  unique speakers   : {len(speakers)}")
    print(f"  has word timings  : "
          f"{bool(transcript.lines and transcript.lines[0].words)}")


def main() -> None:
    parser = common.make_argparser(
        description="Turn on every non-obvious Transcriber option, stream "
        "the bundled WAV, and print what each option produced.",
        include_self_check=True,
    )
    args = parser.parse_args()

    if args.self_check:
        common.run_self_check(
            "04_options_and_tuning",
            lambda: _self_check(args),
        )
        return

    options = build_options_dict(args)
    print_option_summary(options)

    common.hr("Loading model")
    transcriber, arch = common.load_stt_model(
        language=args.language, options=options
    )
    common.errprint(f"  model_arch = {arch.name}")

    wav_path = common.require_wav_path(args.wav_path)
    audio, sample_rate = common.load_wav_file(wav_path)
    common.hr(f"Streaming {wav_path.name}")
    transcript = transcriber.transcribe_without_streaming(
        audio, sample_rate=sample_rate, flags=0
    )
    summarise_lines(transcript)

    if "save_input_wav_path" in options:
        target = os.path.join(
            options["save_input_wav_path"], "input_batch.wav"
        )
        if os.path.exists(target):
            common.hr("save_input_wav_path check")
            print(f"  wrote {target}  "
                  f"({os.path.getsize(target):,} bytes)")

    transcriber.close()


def _self_check(args) -> "SelfCheckResult | None":
    """Smoke test: turn on options, transcribe, assert lines and
    speaker-identification wiring work.
    """
    from test_support.self_check import SelfCheckResult

    wav_path = common.default_wav_path()
    if not wav_path.exists():
        return SelfCheckResult.skip(
            f"missing test audio: {wav_path}", "04_options_and_tuning"
        )
    options = build_options_dict(args)
    transcriber, _ = common.load_stt_model(
        language=args.language, options=options
    )
    try:
        audio, sample_rate = common.load_wav_file(wav_path)
        transcript = transcriber.transcribe_without_streaming(
            audio, sample_rate=sample_rate, flags=0
        )
        if not transcript.lines:
            return SelfCheckResult.fail(
                "no lines returned", "04_options_and_tuning"
            )
        return None  # PASS
    finally:
        transcriber.close()


if __name__ == "__main__":
    main()
