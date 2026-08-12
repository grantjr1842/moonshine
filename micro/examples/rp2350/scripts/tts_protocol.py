"""Shared frame-contract validator for the RP2350 TTS USB protocol.

The Pico firmware speaks a tiny line-based protocol over USB CDC:

    AUDIO <rate> <num_samples>\n    # announces PCM that follows
    <num_samples * 2 bytes little-endian int16>      # the PCM itself
    END <num_samples>\n          # terminator; the count MUST match AUDIO

Both `tts_speak.py` (which issues the protocol) and `usb_audio_bridge.py`
(which receives it) historically re-implemented their own parsers. This
module is the single source of truth: A-102 closed by giving both
scripts the same strict parser with bounded allocations.

The contract enforced here:
  * exactly the documented token count per command (3 for AUDIO, 2 for END)
  * AUDIO rate in [8000, 48000] Hz (matches the C-side envelope bound)
  * num_samples positive, capped at 120 seconds * 48000 Hz = 5_760_000
    (the largest PCM blob we'd ever read into memory; anything bigger
    is a malformed/truncated payload)
  * END count must equal the prior AUDIO count before WAV save or
    playback; mismatches are rejected (A-102 row's "matching terminal
    count" requirement)
  * trailing tokens after the documented arity are a reject
  * non-finite / non-integer / negative values are a reject

Anything that fails returns ``None`` from the parse helper (or raises
``ProtocolError``). Callers MUST treat a ``None``/exception as
"discard the frame"; we never want to allocate or save PCM that came
from a malformed header.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# A-102: hard caps. Defaults match the documented RP2350 protocol; the
# validator is intentionally strict so a malformed command can never
# trigger an unbounded read or a wildly wrong sample rate.
MIN_SAMPLE_RATE_HZ: int = 8000
MAX_SAMPLE_RATE_HZ: int = 48000
# 120 s at 48 kHz = 5_760_000 int16 samples = 11.52 MB. Anything beyond
# is almost certainly a malformed firmware command.
MAX_NUM_SAMPLES: int = 120 * MAX_SAMPLE_RATE_HZ


class ProtocolError(ValueError):
    """Raised when an incoming frame violates the documented contract.

    Callers MUST treat any ProtocolError as "this frame is unrecoverable;
    drop it and resume reading". We never want to allocate or save PCM
    that came from a malformed header.
    """


@dataclass(frozen=True)
class AudioHeader:
    """Parsed ``AUDIO <rate> <num_samples>`` line."""

    rate_hz: int
    num_samples: int


def _check_int(token: str, *, name: str, positive: bool = True) -> int:
    """Strict integer parse: rejects empty, leading/trailing garbage,
    and (optionally) non-positive values. Raises ProtocolError."""
    if not token:
        raise ProtocolError(f"{name}: empty token")
    # str.isdigit() / str.lstrip('-').isdigit() rejects unicode digits,
    # embedded whitespace, plus signs, and exponential notation — all
    # of which an int() call would otherwise accept (with surprising
    # rounding behavior on some inputs). Force strict ASCII-decimal.
    body = token
    if body.startswith("-"):
        if positive:
            raise ProtocolError(f"{name}: negative value not allowed")
        body = body[1:]
    if not body or not body.isascii() or not body.isdigit():
        raise ProtocolError(f"{name}: not an integer: {token!r}")
    return int(body)


def parse_audio_header(line: str) -> AudioHeader:
    """Parse an ``AUDIO <rate> <num_samples>`` line.

    Exactly three whitespace-separated tokens; rate in [8000, 48000];
    num_samples positive and <= MAX_NUM_SAMPLES. Raises ProtocolError
    on any deviation.
    """
    tokens = line.split()
    if len(tokens) != 3:
        raise ProtocolError(
            f"AUDIO: expected 3 tokens, got {len(tokens)}: {line!r}"
        )
    cmd, rate_t, n_t = tokens
    if cmd != "AUDIO":
        raise ProtocolError(f"expected AUDIO header, got {cmd!r}")
    rate = _check_int(rate_t, name="rate")
    if rate < MIN_SAMPLE_RATE_HZ or rate > MAX_SAMPLE_RATE_HZ:
        raise ProtocolError(
            f"rate out of range [{MIN_SAMPLE_RATE_HZ}, "
            f"{MAX_SAMPLE_RATE_HZ}]: {rate}"
        )
    n = _check_int(n_t, name="num_samples")
    if n == 0:
        raise ProtocolError("num_samples must be positive, got 0")
    if n > MAX_NUM_SAMPLES:
        raise ProtocolError(
            f"num_samples exceeds cap {MAX_NUM_SAMPLES}: {n}"
        )
    return AudioHeader(rate_hz=rate, num_samples=n)


def parse_end_frame(line: str, expected_n: int) -> int:
    """Parse an ``END <num_samples>`` line. The count MUST equal
    ``expected_n`` (the AUDIO header's count); a mismatch is a reject.
    Returns the parsed count on success; raises ProtocolError on
    mismatch or malformed input."""
    tokens = line.split()
    if len(tokens) != 2:
        raise ProtocolError(
            f"END: expected 2 tokens, got {len(tokens)}: {line!r}"
        )
    cmd, n_t = tokens
    if cmd != "END":
        raise ProtocolError(f"expected END frame, got {cmd!r}")
    n = _check_int(n_t, name="end_count")
    if n != expected_n:
        raise ProtocolError(
            f"END count {n} != AUDIO count {expected_n}"
        )
    return n


def try_parse_audio_header(line: str) -> Optional[AudioHeader]:
    """Convenience wrapper: returns None on ProtocolError instead of
    raising. Callers that just want to know whether the line is a
    well-formed AUDIO header should use this."""
    try:
        return parse_audio_header(line)
    except ProtocolError:
        return None