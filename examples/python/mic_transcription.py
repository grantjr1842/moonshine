"""Uses the MicTranscriber class to transcribe audio from a microphone."""

# Install the fake sounddevice shim BEFORE any ``moonshine_voice``
# import. ``moonshine_voice.mic_transcriber`` does
# ``import sounddevice as sd`` at module scope, so the fake has
# to be in place before that import fires — otherwise the real
# PortAudio binding wins and the self-check's fake never sees
# the MicTranscriber. The installer is a no-op when
# ``MOONSHINE_SELF_CHECK`` isn't set.
try:
    from test_support import _auto_install  # noqa: F401
except Exception:
    pass

import argparse
import sys
import time

from moonshine_voice import (
    MicTranscriber,
    TranscriptEventListener,
    get_model_for_language,
)


class TerminalListener(TranscriptEventListener):
    def __init__(self):
        self.last_line_text_length = 0

    # Assume we're on a terminal, and so we can use a carriage return to
    # overwrite the last line with the latest text.
    def update_last_terminal_line(self, new_text: str):
        print(f"\r{new_text}", end="", flush=True)
        if len(new_text) < self.last_line_text_length:
            # If the new text is shorter than the last line, we need to
            # overwrite the last line with spaces.
            diff = self.last_line_text_length - len(new_text)
            print(f"{' ' * diff}", end="", flush=True)
        # Update the length of the last line text.
        self.last_line_text_length = len(new_text)

    def on_line_started(self, event):
        self.last_line_text_length = 0

    def on_line_text_changed(self, event):
        self.update_last_terminal_line(event.line.text)

    def on_line_completed(self, event):
        self.update_last_terminal_line(event.line.text)
        print("\n", end="", flush=True)


# If we're not on an interactive terminal, print each line as it's completed.


class FileListener(TranscriptEventListener):
    def on_line_completed(self, event):
        print(event.line.text)


def main():
    parser = argparse.ArgumentParser(description="Basic transcription example")
    parser.add_argument(
        "--language", type=str, default="en", help="Language to use for transcription"
    )
    parser.add_argument(
        "--model-arch",
        type=int,
        default=None,
        help="Model architecture to use for transcription",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run the canned-audio smoke test and exit with PASS/FAIL/SKIP.",
    )
    args = parser.parse_args()

    if args.self_check:
        return _self_check(args)

    model_path, model_arch = get_model_for_language(args.language, args.model_arch)

    mic_transcriber = MicTranscriber(model_path=model_path, model_arch=model_arch)

    if sys.stdout.isatty():
        listener = TerminalListener()
    else:
        listener = FileListener()

    print("Listening to the microphone, press Ctrl+C to stop...", file=sys.stderr)
    mic_transcriber.add_listener(listener)
    mic_transcriber.start()
    try:
        while True:
            time.sleep(0.1)
    finally:
        mic_transcriber.stop()
        mic_transcriber.close()


def _self_check(args):
    """Smoke test: drive MicTranscriber with the fake mic for 10 s."""
    from test_support.self_check import SelfCheckResult, report
    from moonshine_voice import TranscriptEventListener

    class _Counter(TranscriptEventListener):
        def __init__(self):
            super().__init__()
            self.completed = 0

        def on_line_completed(self, event) -> None:
            self.completed += 1

    model_path, model_arch = get_model_for_language(
        args.language, args.model_arch
    )
    mic_transcriber = MicTranscriber(
        model_path=model_path, model_arch=model_arch, samplerate=16000
    )
    counter = _Counter()
    try:
        mic_transcriber.add_listener(counter)
        mic_transcriber.start()
        try:
            # 18 s is enough for the first line to complete
            # (~7-8 s) and gives 2-3 s of headroom. Empirically
            # the first on_line_completed fires around t=7 s.
            time.sleep(18.0)
        finally:
            mic_transcriber.stop()
        if counter.completed < 1:
            report(SelfCheckResult.fail(
                f"no on_line_completed events in 18 s of fake-mic audio",
                "mic_transcription",
            ))
        report(SelfCheckResult.pass_("mic_transcription"))
    finally:
        mic_transcriber.close()


if __name__ == "__main__":
    main()
