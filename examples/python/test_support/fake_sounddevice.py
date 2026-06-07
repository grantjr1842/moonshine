"""A drop-in ``sounddevice.InputStream`` replacement for smoke tests.

The :class:`FakeInputStream` here has the same constructor signature
as the real ``sounddevice.InputStream`` and the same ``start`` /
``stop`` / ``close`` lifecycle. When ``start()`` is called, a
background thread begins feeding a pre-loaded audio buffer through
the ``callback`` every ``blocksize / samplerate`` seconds.

The audio buffer is the bundled ``test-assets/two_cities.wav`` looped
``loop_count`` times (default 3, ≈36 s of audio) — long enough for
any single-segment smoke test.

The shim is installed by :func:`install` which monkey-patches
``sounddevice.InputStream``. The actual swap happens automatically
when the test_support package is loaded under the
``MOONSHINE_SELF_CHECK=1`` environment variable (see
``_auto_install.py``).

Notes
-----
* Only ``InputStream`` is faked. ``OutputStream`` (used by TTS) is
  not — TTS playback is not exercised by the smoke tests. Scripts
  that need TTS (e.g. ``10_full_voice_agent.py``) gate themselves to
  the ``--check-prereqs`` path under ``--self-check``.
* The fake works in two modes: float32 contiguous buffers (the
  default — what the real ``MicTranscriber`` asks for) and int16
  buffers. We mirror the dtype the user requested on
  ``sounddevice.InputStream(..., dtype="float32")``.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

from .canned_audio import load_two_cities, resample


# ``callback`` signature (PortAudio / sounddevice):
#   callback(indata: ndarray, frames: int, time: object, status: object)
# We don't need a real ``CallbackFlags`` / ``CallbackStatus`` —
# constructing an empty object works for our purposes.
class _FakeStatus:
    """Stand-in for ``sd.CallbackFlags``. ``input_underflow`` is
    the only attribute the real code checks; we leave it False."""

    input_underflow = False
    input_overflow = False
    priming_output = False


class FakeInputStream:
    """A drop-in fake of :class:`sounddevice.InputStream`.

    Drives ``self._callback`` from a background thread once
    ``start()`` is called. Audio comes from the cached two_cities
    buffer, looped ``loop_count`` times.

    Parameters mirror the real constructor's keyword arguments so
    that the call site (``mic_transcriber.py:67-68``) doesn't need to
    change.
    """

    def __init__(
        self,
        *,
        samplerate: int,
        blocksize: int = 1024,
        device: Optional[object] = None,
        channels: int = 1,
        dtype: str = "float32",
        callback: Optional[Callable] = None,
        loop_count: int = 3,
        **kwargs,
    ) -> None:
        self._samplerate = int(samplerate)
        self._blocksize = int(blocksize) if blocksize else 1024
        self._channels = int(channels) if channels else 1
        self._dtype = dtype
        self._callback = callback
        self._loop_count = loop_count

        self._audio, file_sr = load_two_cities()
        # The bundled ``two_cities.wav`` is 48 kHz mono. If the
        # caller asked for a different rate we resample on the way
        # in so the callback fires at the requested cadence with
        # the requested rate label.
        if file_sr != self._samplerate:
            self._audio = resample(self._audio, file_sr, self._samplerate)

        self._closed = False
        self._stopped = False
        self._thread: Optional[threading.Thread] = None
        # Cursors / position
        self._cursor = 0
        self._loops_left = loop_count

    # ------------------------------------------------------------------
    # Lifecycle — matches sounddevice.InputStream
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        if self._callback is None:
            return
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run, name="FakeInputStream", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stopped = True
        # Don't join — the test driver will close() and the
        # daemon thread will exit on its own.

    def close(self) -> None:
        self._stopped = True
        self._closed = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # The real class exposes ``read`` / ``streams`` / etc. Tests don't
    # use them; the shim has no need to either.

    # ------------------------------------------------------------------
    # Background pump
    # ------------------------------------------------------------------

    def _next_chunk(self):
        """Return the next ``blocksize`` samples as a buffer of the
        requested dtype and channel layout."""
        n = self._blocksize
        if isinstance(self._audio, list):
            buf = self._audio
            total = len(buf)
            if total == 0:
                return None
            start = self._cursor
            # Handle the wrap-around at the loop boundary.
            if start + n <= total:
                chunk = buf[start : start + n]
                self._cursor = (start + n) % total
                if self._cursor == 0:
                    self._loops_left -= 1
            else:
                # Wrap: take from start to end, then from 0 onward.
                tail = buf[start:]
                self._cursor = n - len(tail)
                head = buf[: self._cursor]
                chunk = tail + head
                self._loops_left -= 1
        else:
            # numpy path
            import numpy as np  # type: ignore

            buf = self._audio
            total = buf.shape[0]
            if total == 0:
                return None
            start = self._cursor
            end = start + n
            if end <= total:
                chunk = buf[start:end].copy()
                self._cursor = end
                if self._cursor == total:
                    self._cursor = 0
                    self._loops_left -= 1
            else:
                # Wrap around the loop boundary.
                tail = buf[start:].copy()
                head = buf[: n - tail.shape[0]].copy()
                chunk = np.concatenate([tail, head])
                self._cursor = head.shape[0]
                self._loops_left -= 1

        # Shape to (blocksize, channels) — sounddevice delivers frames
        # in rows, even for mono.
        if self._channels > 1 and isinstance(chunk, list):
            chunk = [
                sample
                for sample in chunk
                for _ in range(self._channels)
            ]
        elif self._channels > 1:
            import numpy as np  # type: ignore
            chunk = np.repeat(chunk[:, None], self._channels, axis=1)

        # Cast dtype if needed. sounddevice delivers float32 by
        # default; int16 only if the user asked. We keep float32
        # for both since the C library's add_audio wants float.
        if self._dtype in ("int16", "<i2") and not isinstance(chunk, list):
            import numpy as np  # type: ignore
            chunk = (chunk * 32768.0).astype("<i2")
        return chunk

    def _run(self) -> None:
        """Pump chunks into the callback at the right wall-clock rate."""
        try:
            import numpy as np  # type: ignore
        except ImportError:
            np = None  # type: ignore

        period = self._blocksize / float(self._samplerate)
        # Drive the callback as fast as the test driver expects
        # events. The C library's VAD pipeline doesn't care about
        # wall-clock precision — it just needs callbacks fired
        # faster than its own 0.5 s update tick. We pace at 1.1x
        # the natural period for a small safety margin.
        while not self._stopped and self._loops_left > 0:
            chunk = self._next_chunk()
            if chunk is None:
                break
            try:
                if np is not None and not isinstance(chunk, list):
                    self._callback(chunk, self._blocksize, None, _FakeStatus)
                else:
                    # Wrap list chunk as ndarray if numpy is available.
                    if np is not None:
                        import numpy as _np
                        arr = _np.asarray(chunk, dtype="float32")
                        if arr.ndim == 1:
                            arr = arr.reshape(-1, self._channels)
                        self._callback(
                            arr, self._blocksize, None, _FakeStatus
                        )
                    else:
                        # No numpy — pass the list directly. This
                        # path only works for callbacks that don't
                        # care about dtype.
                        self._callback(
                            chunk, self._blocksize, None, _FakeStatus
                        )
            except Exception:
                # Don't let a misbehaving callback kill the thread.
                pass
            time.sleep(period * 0.9)


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------

_INSTALLED = False
_ORIGINAL = None  # type: ignore


def install(*, loop_count: int = 3) -> None:
    """Replace :class:`sounddevice.InputStream` with :class:`FakeInputStream`.

    Idempotent — calling ``install`` twice is a no-op. The shim is
    process-wide: any ``MicTranscriber`` constructed after this call
    will see the fake.

    ``loop_count`` controls how many times the canned audio is
    replayed before the stream goes silent. The default (3) gives
    ≈36 s of audio — enough for any single-segment smoke test.
    """
    global _INSTALLED, _ORIGINAL
    if _INSTALLED:
        return
    import sounddevice  # type: ignore

    _ORIGINAL = sounddevice.InputStream

    class _BoundFakeInputStream(FakeInputStream):
        """A subclass that pins ``loop_count`` at install time."""

        def __init__(self, **kwargs):
            kwargs.setdefault("loop_count", loop_count)
            super().__init__(**kwargs)

    sounddevice.InputStream = _BoundFakeInputStream  # type: ignore
    _INSTALLED = True


def uninstall() -> None:
    """Restore the original ``sounddevice.InputStream``.

    Primarily for tests of the test support itself; production
    smoke tests don't need to call this — the monkey-patch lives
    for the lifetime of the process and that's fine.
    """
    global _INSTALLED, _ORIGINAL
    if not _INSTALLED:
        return
    import sounddevice  # type: ignore

    sounddevice.InputStream = _ORIGINAL  # type: ignore
    _INSTALLED = False
    _ORIGINAL = None
