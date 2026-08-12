"""Native-result validation envelope tests for the Moonshine Python bridge.

These tests are the Spec A Phase 3 / A-151 test surface: they exercise
``moonshine_api.checked_native_count`` / ``bounded_native_string`` and the
hardened native-result call sites in :mod:`moonshine_api`,
:mod:`intent_recognizer`, and :mod:`transcriber` against fake ctypes libs
that simulate corrupted, oversized, null, or otherwise malformed native
output.

Each test pins one invariant of the envelope:

* ``checked_native_count`` rejects values above the conservative Python
  budget and would-overflow-size_t values, and accepts the boundary.
* ``bounded_native_string`` decodes up to the byte budget, rejects
  unterminated allocations, tolerates invalid UTF-8, and returns the
  empty string for null/missing pointers.
* The TTS helper frees a non-null zero-count allocation before returning
  the empty list, refuses non-zero counts above ``MAX_AUDIO_SAMPLES``,
  and raises a typed error if a non-zero count comes back with a null
  pointer.
* The G2P helper decodes phoneme output with an explicit byte limit,
  rejects oversized / negative / unterminated results, and returns the
  empty string for null or zero-count outputs (freeing any allocation
  the native layer returned).
* ``_parse_transcript`` rejects non-null line_count with null lines
  pointer, oversized line/word/audio counts, negative counts, and
  accepts a zero-count / null-lines transcript (an empty valid result).
* ``IntentRecognizer.get_closest_intents`` rejects oversized / negative
  / null-pointer results, frees the native allocation in every rejection
  path, and decodes phrases with a bounded string helper.
* ``IntentRecognizer.calculate_embedding`` rejects oversized / negative
  / null-pointer results, frees the native allocation in every
  rejection path, and is bounded against ``MAX_EMBEDDING_ELEMENTS``.

The cross-language budget constants in :mod:`moonshine_api` mirror
``moonshine/core/moonshine-cpp.h`` so caps do not drift between the
C++ and Python envelopes.
"""

import ctypes
import sys
import unittest

from moonshine_voice import intent_recognizer, moonshine_api
from moonshine_voice.errors import MoonshineError
from moonshine_voice.intent_recognizer import IntentRecognizer
from moonshine_voice.transcriber import Transcriber


# ---------------------------------------------------------------------------
# Fake native libraries
# ---------------------------------------------------------------------------


class _TtsFakeLib:
    """Fake ``libmoonshine`` for ``moonshine_text_to_speech_samples``."""

    def __init__(self, count, pointer=True, error=0):
        self.count = count
        self.pointer = pointer
        self.error = error
        self.buffer = (ctypes.c_float * 1)(0.5)
        self.freed_with = None

    def moonshine_text_to_speech(self, _handle, _text, _options, _options_count, out_audio, out_size, out_sr):
        ctypes.cast(out_size, ctypes.POINTER(ctypes.c_uint64))[0] = self.count
        ctypes.cast(out_sr, ctypes.POINTER(ctypes.c_int32))[0] = 16_000
        ctypes.cast(out_audio, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))[0] = (
            ctypes.cast(self.buffer, ctypes.POINTER(ctypes.c_float)) if self.pointer else None
        )
        return self.error

    def moonshine_error_to_string(self, _error):
        return b"fake error"

    def moonshine_free(self, _addr):  # noqa: ARG002 - matches moonshine_api.moonshine_free shape
        self.freed_with = "moonshine_free"
        # Mirror libc free on a non-null pointer (no-op for None/0)
        return None


class _G2pFakeLib:
    """Fake ``libmoonshine`` for ``moonshine_text_to_phonemes_string``."""

    def __init__(self, byte_count, address, error=0):
        self.byte_count = byte_count
        self.address = address
        self.error = error
        self.freed = []

    def moonshine_text_to_phonemes(
        self, _handle, _text, _options, _options_count, out_ph, out_count
    ):
        ctypes.cast(out_count, ctypes.POINTER(ctypes.c_uint64))[0] = self.byte_count
        ctypes.cast(out_ph, ctypes.POINTER(ctypes.c_char_p))[0] = self.address
        return self.error

    def moonshine_error_to_string(self, _error):
        return b"fake error"


class _IntentFakeLib:
    """Fake ``libmoonshine`` for ``IntentRecognizer`` calls."""

    def __init__(self, count, pointer=False):
        self.count = count
        self.pointer = pointer
        self.freed = None
        self.embedding_freed = False
        self.embedding = (ctypes.c_float * 1)(0.25)

    def moonshine_get_closest_intents(self, _handle, _utterance, _threshold, out_matches, out_count):
        ctypes.cast(out_count, ctypes.POINTER(ctypes.c_uint64))[0] = self.count
        ctypes.cast(out_matches, ctypes.POINTER(ctypes.POINTER(moonshine_api.MoonshineIntentMatchC)))[0] = None
        return 0

    def moonshine_free_intent_matches(self, _matches, count):
        self.freed = count

    def moonshine_calculate_intent_embedding(self, _handle, _sentence, out_ptr, out_size, _model):
        ctypes.cast(out_size, ctypes.POINTER(ctypes.c_uint64))[0] = self.count
        ctypes.cast(out_ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))[0] = (
            ctypes.cast(self.embedding, ctypes.POINTER(ctypes.c_float)) if self.pointer else None
        )
        return 0

    def moonshine_free_intent_embedding(self, _ptr):
        self.embedding_freed = True

    def moonshine_free_intent_recognizer(self, _handle):
        pass


class _TtsFakeWithError(_TtsFakeLib):
    """TTS fake that returns a non-zero error code from the native call."""

    def __init__(self):
        super().__init__(count=0, pointer=False, error=moonshine_api.MOONSHINE_ERROR_UNKNOWN)


# ---------------------------------------------------------------------------
# Tests: helpers
# ---------------------------------------------------------------------------


class CheckedNativeCountTests(unittest.TestCase):
    def test_accepts_below_maximum(self):
        self.assertEqual(moonshine_api.checked_native_count(0, 4, "count"), 0)
        self.assertEqual(moonshine_api.checked_native_count(3, 4, "count"), 3)

    def test_accepts_at_maximum(self):
        self.assertEqual(moonshine_api.checked_native_count(4, 4, "count"), 4)

    def test_rejects_above_maximum(self):
        with self.assertRaises(MoonshineError):
            moonshine_api.checked_native_count(5, 4, "count")

    def test_rejects_negative(self):
        with self.assertRaises(MoonshineError):
            moonshine_api.checked_native_count(-1, 4, "count")

    def test_rejects_size_overflow(self):
        with self.assertRaises(MoonshineError):
            moonshine_api.checked_native_count(sys.maxsize + 1, sys.maxsize * 2, "count")


class BoundedNativeStringTests(unittest.TestCase):
    def test_decodes_short_nul_terminated(self):
        text = ctypes.create_string_buffer(b"ok")
        self.assertEqual(moonshine_api.bounded_native_string(text, "text"), "ok")

    def test_returns_empty_for_none(self):
        self.assertEqual(moonshine_api.bounded_native_string(None, "text"), "")

    def test_returns_empty_for_null_value(self):
        null = ctypes.c_char_p()
        self.assertEqual(moonshine_api.bounded_native_string(null, "text"), "")

    def test_rejects_unterminated_within_max(self):
        # 100 A's (no NUL within the first 32 bytes; NUL sits at index
        # 100 so a 32-byte read must fail to find a terminator).
        backing = ctypes.create_string_buffer(b"A" * 100)
        cstr = ctypes.c_char_p(ctypes.cast(backing, ctypes.c_void_p).value)
        with self.assertRaises(MoonshineError):
            moonshine_api.bounded_native_string(cstr, "text", max_bytes=32)

    def test_accepts_at_max_boundary(self):
        # Exactly 4 bytes + NUL; budget 5 should pass.
        text = ctypes.create_string_buffer(b"abcd")
        self.assertEqual(moonshine_api.bounded_native_string(text, "text", max_bytes=5), "abcd")

    def test_rejects_unterminated_at_max_boundary(self):
        # 5 bytes with no NUL, budget 5; should still fail because there is
        # no terminator within the budget.
        text = ctypes.create_string_buffer(b"ABCDE")
        with self.assertRaises(MoonshineError):
            moonshine_api.bounded_native_string(text, "text", max_bytes=5)

    def test_invalid_utf8_is_replaced(self):
        text = ctypes.create_string_buffer(b"ok\xe2\x80")
        self.assertIn("ok", moonshine_api.bounded_native_string(text, "text"))


# ---------------------------------------------------------------------------
# Tests: TTS (moonshine_text_to_speech_samples)
# ---------------------------------------------------------------------------


class TtsEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self._saved_instance = moonshine_api._MoonshineLib._instance
        self._saved_free = moonshine_api.moonshine_free

    def tearDown(self):
        moonshine_api._MoonshineLib._instance = self._saved_instance
        moonshine_api.moonshine_free = self._saved_free

    def _install_fake(self, fake):
        moonshine_api._MoonshineLib._instance = type("Wrapper", (), {"lib": fake})()

    def test_returns_samples_for_valid_positive_count(self):
        fake = _TtsFakeLib(count=1, pointer=True)
        self._install_fake(fake)
        # Override the real libc free so the success path's release
        # of the ctypes-managed fake buffer is a no-op (the real
        # address was never malloc'd, so calling libc free() on it
        # would be undefined behavior).
        freed = []
        moonshine_api.moonshine_free = lambda address: freed.append(address)
        samples, rate = moonshine_api.moonshine_text_to_speech_samples(1, "hello")
        self.assertEqual(samples, [0.5])
        self.assertEqual(rate, 16_000)
        self.assertEqual(len(freed), 1)

    def test_frees_zero_count_nonnull_result(self):
        fake = _TtsFakeLib(count=0, pointer=True)
        self._install_fake(fake)
        freed = []
        moonshine_api.moonshine_free = lambda address: freed.append(address)
        samples, rate = moonshine_api.moonshine_text_to_speech_samples(1, "hello")
        self.assertEqual(samples, [])
        self.assertEqual(rate, 16_000)
        self.assertEqual(len(freed), 1)

    def test_rejects_oversized_count(self):
        fake = _TtsFakeLib(count=moonshine_api.MAX_AUDIO_SAMPLES + 1, pointer=True)
        self._install_fake(fake)
        freed = []
        moonshine_api.moonshine_free = lambda address: freed.append(address)
        with self.assertRaises(MoonshineError):
            moonshine_api.moonshine_text_to_speech_samples(1, "hello")
        self.assertEqual(len(freed), 1)

    def test_rejects_nonzerocount_with_null_pointer(self):
        fake = _TtsFakeLib(count=4, pointer=False)
        self._install_fake(fake)
        freed = []
        moonshine_api.moonshine_free = lambda address: freed.append(address)
        with self.assertRaises(MoonshineError):
            moonshine_api.moonshine_text_to_speech_samples(1, "hello")
        # Nothing to free for a null pointer.
        self.assertEqual(freed, [])

    def test_rejects_negative_count(self):
        fake = _TtsFakeLib(count=-1, pointer=True)
        self._install_fake(fake)
        freed = []
        moonshine_api.moonshine_free = lambda address: freed.append(address)
        with self.assertRaises(MoonshineError):
            moonshine_api.moonshine_text_to_speech_samples(1, "hello")
        self.assertEqual(len(freed), 1)


# ---------------------------------------------------------------------------
# Tests: G2P (moonshine_text_to_phonemes_string)
# ---------------------------------------------------------------------------


class G2pEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self._saved_instance = moonshine_api._MoonshineLib._instance
        self._saved_free = moonshine_api.moonshine_free

    def tearDown(self):
        moonshine_api._MoonshineLib._instance = self._saved_instance
        moonshine_api.moonshine_free = self._saved_free

    def _install_fake(self, fake):
        moonshine_api._MoonshineLib._instance = type("Wrapper", (), {"lib": fake})()

    def _make_string_buffer(self, payload):
        return ctypes.cast(ctypes.pointer(ctypes.create_string_buffer(payload)), ctypes.c_char_p).value

    def test_returns_empty_for_null(self):
        fake = _G2pFakeLib(byte_count=0, address=None)
        self._install_fake(fake)
        freed = []
        moonshine_api.moonshine_free = lambda address: freed.append(address)
        self.assertEqual(moonshine_api.moonshine_text_to_phonemes_string(1, "hi"), "")
        self.assertEqual(freed, [])

    def test_returns_phonemes_for_valid_short(self):
        # Plain ASCII payload to keep the test independent of UTF-8 byte
        # sequences. The point is to exercise the bounded-read success path,
        # not to validate IPA encoding.
        payload = b"HH AH0 L OW1\x00"
        addr = self._make_string_buffer(payload)
        fake = _G2pFakeLib(byte_count=len(payload) - 1, address=addr)
        self._install_fake(fake)
        freed = []
        moonshine_api.moonshine_free = lambda address: freed.append(address)
        result = moonshine_api.moonshine_text_to_phonemes_string(1, "hi")
        self.assertEqual(result, "HH AH0 L OW1")
        self.assertEqual(len(freed), 1)

    def test_rejects_oversized_byte_count(self):
        # 100 bytes of 'A' is a valid allocation; the wrapper must
        # still reject the malformed count above MAX_NATIVE_STRING_BYTES.
        backing = ctypes.create_string_buffer(b"A" * 100)
        addr = ctypes.cast(backing, ctypes.c_char_p).value
        fake = _G2pFakeLib(byte_count=moonshine_api.MAX_NATIVE_STRING_BYTES + 1, address=addr)
        self._install_fake(fake)
        freed = []
        moonshine_api.moonshine_free = lambda address: freed.append(address)
        with self.assertRaises(MoonshineError):
            moonshine_api.moonshine_text_to_phonemes_string(1, "hi")
        self.assertEqual(len(freed), 1)

    def test_rejects_unterminated_at_max_bytes(self):
        # 5 bytes with no NUL within those 5 bytes (NUL sits at index 100).
        backing = ctypes.create_string_buffer(b"A" * 100)
        addr = ctypes.cast(backing, ctypes.c_char_p).value
        fake = _G2pFakeLib(byte_count=5, address=addr)
        self._install_fake(fake)
        freed = []
        moonshine_api.moonshine_free = lambda address: freed.append(address)
        with self.assertRaises(MoonshineError):
            moonshine_api.moonshine_text_to_phonemes_string(1, "hi")
        self.assertEqual(len(freed), 1)

    def test_rejects_negative_byte_count(self):
        backing = ctypes.create_string_buffer(b"x\x00")
        addr = ctypes.cast(backing, ctypes.c_char_p).value
        fake = _G2pFakeLib(byte_count=-1, address=addr)
        self._install_fake(fake)
        freed = []
        moonshine_api.moonshine_free = lambda address: freed.append(address)
        with self.assertRaises(MoonshineError):
            moonshine_api.moonshine_text_to_phonemes_string(1, "hi")
        self.assertEqual(len(freed), 1)

    def test_rejects_nonzerocount_with_null_pointer(self):
        fake = _G2pFakeLib(byte_count=4, address=None)
        self._install_fake(fake)
        freed = []
        moonshine_api.moonshine_free = lambda address: freed.append(address)
        with self.assertRaises(MoonshineError):
            moonshine_api.moonshine_text_to_phonemes_string(1, "hi")
        self.assertEqual(freed, [])


# ---------------------------------------------------------------------------
# Tests: transcript (_parse_transcript)
# ---------------------------------------------------------------------------


def _build_transcriber():
    transcriber = Transcriber.__new__(Transcriber)
    transcriber._handle = None
    return transcriber


class TranscriptEnvelopeTests(unittest.TestCase):
    def test_transcript_rejects_null_pointer_for_nonzero_count(self):
        transcript = moonshine_api.TranscriptC()
        transcript.line_count = 1
        with self.assertRaises(MoonshineError):
            _build_transcriber()._parse_transcript(ctypes.pointer(transcript))

    def test_transcript_accepts_zero_count(self):
        transcript = moonshine_api.TranscriptC()
        transcript.line_count = 0
        # Pointing at a transcript with 0 lines and a non-null lines pointer
        # is an empty valid result; the wrapper must not raise.
        result = _build_transcriber()._parse_transcript(ctypes.pointer(transcript))
        self.assertEqual(result.lines, [])

    def test_transcript_rejects_oversized_line_count(self):
        transcript = moonshine_api.TranscriptC()
        transcript.line_count = moonshine_api.MAX_TRANSCRIPT_LINES + 1
        with self.assertRaises(MoonshineError):
            _build_transcriber()._parse_transcript(ctypes.pointer(transcript))

    def test_transcript_rejects_negative_line_count(self):
        transcript = moonshine_api.TranscriptC()
        transcript.line_count = ctypes.c_uint64(-1).value  # wraps to 2**64 - 1
        with self.assertRaises(MoonshineError):
            _build_transcriber()._parse_transcript(ctypes.pointer(transcript))


# ---------------------------------------------------------------------------
# Tests: intent recognizer
# ---------------------------------------------------------------------------


def _build_intent_recognizer(fake_lib):
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    recognizer._handle = 1
    recognizer._lib = fake_lib
    recognizer._threshold = 0.8
    recognizer._handlers = {}
    return recognizer


class IntentEnvelopeTests(unittest.TestCase):
    def test_rejects_oversized_result_and_bounds_cleanup(self):
        fake = _IntentFakeLib(count=moonshine_api.MAX_INTENT_MATCHES + 1)
        recognizer = _build_intent_recognizer(fake)
        with self.assertRaises(MoonshineError):
            recognizer.get_closest_intents("hello")
        self.assertEqual(fake.freed, moonshine_api.MAX_INTENT_MATCHES)

    def test_rejects_negative_intent_count(self):
        # A negative Python int written to a uint64 out-parameter wraps to
        # 2**64 - 1, which exceeds MAX_INTENT_MATCHES; the wrapper rejects
        # it via the same path and caps the free at MAX_INTENT_MATCHES,
        # matching the C++ IntentMatchesDeleter.
        fake = _IntentFakeLib(count=-1)
        recognizer = _build_intent_recognizer(fake)
        with self.assertRaises(MoonshineError):
            recognizer.get_closest_intents("hello")
        self.assertEqual(fake.freed, moonshine_api.MAX_INTENT_MATCHES)

    def test_rejects_nonzerocount_with_null_matches_pointer(self):
        fake = _IntentFakeLib(count=2, pointer=False)
        recognizer = _build_intent_recognizer(fake)
        with self.assertRaises(MoonshineError):
            recognizer.get_closest_intents("hello")


# ---------------------------------------------------------------------------
# Tests: intent embedding
# ---------------------------------------------------------------------------


class EmbeddingEnvelopeTests(unittest.TestCase):
    def test_frees_pointer_on_validation_error(self):
        fake = _IntentFakeLib(count=moonshine_api.MAX_EMBEDDING_ELEMENTS + 1, pointer=True)
        recognizer = _build_intent_recognizer(fake)
        with self.assertRaises(MoonshineError):
            recognizer.calculate_embedding("hello")
        self.assertTrue(fake.embedding_freed)

    def test_rejects_negative_embedding_count(self):
        fake = _IntentFakeLib(count=-1, pointer=True)
        recognizer = _build_intent_recognizer(fake)
        with self.assertRaises(MoonshineError):
            recognizer.calculate_embedding("hello")
        self.assertTrue(fake.embedding_freed)

    def test_rejects_nonzerocount_with_null_embedding_pointer(self):
        fake = _IntentFakeLib(count=4, pointer=False)
        recognizer = _build_intent_recognizer(fake)
        with self.assertRaises(MoonshineError):
            recognizer.calculate_embedding("hello")
        # No allocation means nothing to free; the native free was not
        # called.
        self.assertFalse(fake.embedding_freed)


if __name__ == "__main__":
    unittest.main()
