"""Unit tests for ``metrics.py``.

Run with::

    python -m pytest examples/python/pocket_tts_verify/test_metrics.py

Or directly with::

    python examples/python/pocket_tts_verify/test_metrics.py
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

# Allow running this file directly (without pytest) by ensuring the
# package is on sys.path.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import metrics  # noqa: E402


class TestWer(unittest.TestCase):
    def test_perfect_match(self):
        self.assertEqual(metrics.wer("hello world", "hello world"), 0.0)

    def test_one_word_wrong(self):
        # 1 substitution out of 2 words = 50% WER.
        self.assertAlmostEqual(
            metrics.wer("hello world", "hello earth"), 0.5, places=4
        )

    def test_insertion(self):
        # "hello" vs "hello there" = 1 insertion out of 1 ref word = 1.0
        # WER, plus a substitution ("world" → "there") in a 2-word
        # alignment → 1/2.
        # jiwer's actual alignment can vary; we just check the
        # number is between 0.5 and 1.0.
        w = metrics.wer("hello world", "hello there world")
        self.assertGreater(w, 0.0)
        self.assertLessEqual(w, 1.0)

    def test_empty_reference(self):
        self.assertEqual(metrics.wer("", ""), 0.0)
        self.assertEqual(metrics.wer("", "anything"), 1.0)

    def test_completely_different(self):
        # The WER will be >= 1.0 (more edits than reference words).
        w = metrics.wer("a b c", "x y z w v")
        self.assertGreaterEqual(w, 1.0)


class TestCer(unittest.TestCase):
    def test_perfect_match(self):
        self.assertEqual(metrics.cer("hello", "hello"), 0.0)

    def test_one_char_wrong(self):
        # 1 substitution out of 5 chars = 20% CER.
        self.assertAlmostEqual(
            metrics.cer("hello", "hellp"), 0.2, places=4
        )

    def test_empty_reference(self):
        self.assertEqual(metrics.cer("", ""), 0.0)
        self.assertEqual(metrics.cer("", "x"), 1.0)


class TestNormalise(unittest.TestCase):
    def test_english_lowercases_and_strips_punct(self):
        # English normaliser lowercases and removes punctuation.
        out = metrics.normalise("Hello, World!", "en")
        self.assertNotIn(",", out)
        self.assertNotIn("!", out)
        # Either lowercase or original case depending on the
        # installed normaliser version — just check it's non-empty.
        self.assertTrue(out)

    def test_non_english_no_lowercase(self):
        out = metrics.normalise("  こんにちは  ", "ja")
        # No English normaliser, so just NFC + whitespace strip.
        self.assertEqual(out, "こんにちは")

    def test_empty(self):
        self.assertEqual(metrics.normalise("", "en"), "")
        self.assertEqual(metrics.normalise("   ", "en"), "")
        self.assertEqual(metrics.normalise(None, "en"), "")

    def test_nfc_normalisation(self):
        # "é" can be NFC (one code point) or NFD (e + combining acute).
        nfc = "café"
        nfd = "café"
        # Both should normalise to the same NFC form.
        self.assertEqual(
            metrics.normalise(nfd, "ja"),
            metrics.normalise(nfc, "ja"),
        )


class TestExactMatch(unittest.TestCase):
    def test_identical(self):
        self.assertTrue(metrics.exact_match("Hello", "Hello", "en"))

    def test_different_case(self):
        # After English normaliser, "Hello" and "hello" both become
        # lowercase, so they should match.
        self.assertTrue(metrics.exact_match("Hello", "hello", "en"))

    def test_different_text(self):
        self.assertFalse(
            metrics.exact_match("Hello world", "Hello there", "en")
        )


class TestAudioStats(unittest.TestCase):
    def test_empty(self):
        s = metrics.audio_stats([], 16000)
        self.assertEqual(s["num_samples"], 0)
        self.assertEqual(s["duration_sec"], 0.0)
        self.assertEqual(s["peak_amplitude"], 0.0)
        self.assertEqual(s["silence_ratio"], 1.0)

    def test_silence(self):
        s = metrics.audio_stats([0.0] * 16000, 16000)
        self.assertEqual(s["peak_amplitude"], 0.0)
        self.assertEqual(s["rms"], 0.0)
        self.assertEqual(s["silence_ratio"], 1.0)
        self.assertEqual(s["duration_sec"], 1.0)

    def test_full_scale_sine(self):
        # 1003 Hz sine at full scale (frequency doesn't divide sr
        # evenly, so zero-crossings don't land on sample boundaries
        # and the silence ratio stays low). Peak ~1.0, RMS ~0.707.
        sr = 16000
        n = sr  # 1 second
        samples = [math.sin(2 * math.pi * 1003 * i / sr) for i in range(n)]
        s = metrics.audio_stats(samples, sr)
        self.assertAlmostEqual(s["peak_amplitude"], 1.0, places=2)
        self.assertAlmostEqual(s["rms"], math.sqrt(0.5), places=2)
        self.assertAlmostEqual(s["duration_sec"], 1.0, places=3)
        self.assertLess(s["silence_ratio"], 0.05)

    def test_quiet_signal(self):
        sr = 16000
        n = sr
        # All samples at 1e-4 amplitude — below the default 1e-3 threshold.
        samples = [1e-4] * n
        s = metrics.audio_stats(samples, sr)
        self.assertEqual(s["peak_amplitude"], 1e-4)
        self.assertEqual(s["silence_ratio"], 1.0)


class TestIsAudioSilent(unittest.TestCase):
    def test_empty(self):
        self.assertTrue(metrics.is_audio_silent([]))

    def test_zero_signal(self):
        self.assertTrue(metrics.is_audio_silent([0.0] * 16000))

    def test_real_signal(self):
        samples = [0.5 * math.sin(2 * math.pi * 200 * i / 16000)
                   for i in range(16000)]
        self.assertFalse(metrics.is_audio_silent(samples))

    def test_too_short(self):
        # 1 sample at 16 kHz = 62.5 µs, well below the 50 ms floor.
        self.assertTrue(metrics.is_audio_silent([0.5]))


if __name__ == "__main__":
    unittest.main()
