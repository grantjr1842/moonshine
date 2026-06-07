"""Load ``test-assets/two_cities.wav`` as float32 mono at 16 kHz.

The bundled ``two_cities.wav`` is what every other test in the repo
uses, so we reuse it here. The function returns a NumPy ``float32``
array and the sample rate, ready to feed straight into a
``FakeInputStream`` callback.

If ``numpy`` is not importable (it is a hard dependency for the
transcriber in any realistic setup, but a strict minimum install
might omit it) the loader falls back to a plain Python ``list[float]``
and a sample rate of 16000. The fake stream accepts both forms.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Union

# The repo's test-assets directory. We honour the same
# ``MOONSHINE_TEST_ASSETS_DIR`` env var that
# ``examples/python/stt/common.py`` uses, and fall back to a path
# derived from this file's location (so the support package works
# whether the test is run from the repo root or from a tmpdir).
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _two_cities_path() -> Path:
    """Resolve the path to ``two_cities.wav``.

    Order of precedence:
      1. ``MOONSHINE_TEST_ASSETS_DIR`` env var (lets the driver
         point us at a custom location — useful in CI where the
         repo is mounted under a different prefix).
      2. ``<repo>/test-assets/two_cities.wav`` based on this file's
         position in the tree.
    """
    env_dir = os.environ.get("MOONSHINE_TEST_ASSETS_DIR")
    if env_dir:
        return Path(env_dir) / "two_cities.wav"
    return _REPO_ROOT / "test-assets" / "two_cities.wav"


# Module-level cache so the file is only read once per process.
# ``FakeInputStream`` instances all share the same underlying buffer.
_AUDIO_CACHE: dict = {}


def load_two_cities() -> Tuple[Union["list", "object"], int]:
    """Return ``(samples, sample_rate)`` for the bundled two_cities.wav.

    Returns a NumPy ``float32`` array and the file's sample rate when
    ``numpy`` is available; otherwise returns a Python ``list[float]``
    and ``16000``. Multi-channel files are downmixed to mono.

    Raises ``FileNotFoundError`` with a clear message if the WAV is
    missing — that is the most common failure on a fresh checkout
    where the LFS-tracked test fixtures haven't been pulled yet.
    """
    if "two_cities" in _AUDIO_CACHE:
        return _AUDIO_CACHE["two_cities"]

    path = _two_cities_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Bundled test audio not found: {path}\n"
            "Pull the LFS-tracked test fixtures (test-assets/two_cities.wav) "
            "or set MOONSHINE_TEST_ASSETS_DIR to a directory containing the file."
        )

    try:
        import numpy as np  # type: ignore
    except ImportError:
        np = None  # type: ignore

    # Read the WAV with the stdlib ``wave`` module so we don't pull in
    # scipy / soundfile as a hard dep. two_cities.wav is 16-bit PCM.
    import wave

    with wave.open(str(path), "rb") as w:
        n_channels = w.getnchannels()
        sample_rate = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)

    if np is not None:
        # ``np.frombuffer`` views the bytes as int16, then we normalise
        # to float32 in [-1.0, 1.0] — the same range the C library
        # expects on add_audio().
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if n_channels > 1:
            # Downmix to mono by averaging channels.
            audio = audio.reshape(-1, n_channels).mean(axis=1)
    else:
        import struct

        n_samples = len(raw) // 2  # 16-bit
        samples = list(struct.unpack(f"<{n_samples}h", raw))
        audio = [s / 32768.0 for s in samples]
        if n_channels > 1:
            audio = [
                sum(audio[i::n_channels]) / n_channels
                for i in range(n_channels)
            ]

    _AUDIO_CACHE["two_cities"] = (audio, sample_rate)
    return _AUDIO_CACHE["two_cities"]


def resample(audio, src_rate: int, dst_rate: int):
    """Resample ``audio`` from ``src_rate`` to ``dst_rate``.

    Uses linear interpolation — adequate for a smoke test where we
    just need recognisable speech to hit the VAD. Doesn't introduce
    a scipy / librosa dependency.

    Accepts both ``ndarray`` and ``list``; returns the same type.
    """
    if src_rate == dst_rate or len(audio) < 2:
        return audio
    if isinstance(audio, list):
        # Pure-Python linear interpolation
        n_src = len(audio)
        # Output length scales with the rate ratio.
        n_dst = max(1, int(round(n_src * dst_rate / src_rate)))
        out = [0.0] * n_dst
        for i in range(n_dst):
            # Map i in [0, n_dst) to a float position in [0, n_src - 1].
            pos = i * (n_src - 1) / (n_dst - 1) if n_dst > 1 else 0.0
            lo = int(pos)
            hi = min(lo + 1, n_src - 1)
            frac = pos - lo
            out[i] = audio[lo] * (1.0 - frac) + audio[hi] * frac
        return out
    import numpy as np  # type: ignore

    n_src = audio.shape[0]
    n_dst = max(1, int(round(n_src * dst_rate / src_rate)))
    if n_dst == 1:
        return audio[:1].copy()
    x_old = np.linspace(0.0, 1.0, n_src, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, n_dst, dtype=np.float64)
    return np.interp(x_new, x_old, audio.astype(np.float64)).astype(np.float32)
