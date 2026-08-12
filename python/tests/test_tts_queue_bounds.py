import queue
import unittest

from moonshine_voice.errors import MoonshineError
from moonshine_voice.tts import (
    MAX_SAY_QUEUE_ITEMS,
    TextToSpeech,
    _SayRequest,
)


class TtsQueueBoundTests(unittest.TestCase):
    def setUp(self):
        self.tts = TextToSpeech.__new__(TextToSpeech)
        self.tts._closed = False
        self.tts._say_queue = queue.Queue(maxsize=MAX_SAY_QUEUE_ITEMS)

    def test_queue_rejects_overflow(self):
        request = _SayRequest("hello", None, None, None, None)
        for _ in range(MAX_SAY_QUEUE_ITEMS):
            self.tts._enqueue_say_request(request)
        with self.assertRaises(MoonshineError):
            self.tts._enqueue_say_request(request)

    def test_queue_rejects_oversized_text(self):
        request = _SayRequest("x" * (1_048_576 + 1), None, None, None, None)
        with self.assertRaises(MoonshineError):
            self.tts._enqueue_say_request(request)


if __name__ == "__main__":
    unittest.main()
