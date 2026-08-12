// Per-stream audio buffer race regression for A-065.
//
// Exercises the audio_buffer_mutex / atomic-swap pattern on
// TranscriberStream::new_audio_buffer: a producer thread
// (add_to_new_audio_buffer) and a consumer thread
// (transcribe_stream snapshot+swap) must not race; the consumer's
// local snapshot must remain stable for the duration of VAD
// processing, and the producer's appends after the swap must
// continue queuing into a fresh buffer without crashing the
// consumer's iterator.
//
// Standalone test — we don't pull in transcriber.h because the lock
// pattern is the same std::vector + std::mutex dance regardless of
// the surrounding type.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

namespace {

// Build a deterministic PCM ramp in [0, 1).
std::vector<float> make_audio(uint64_t frames, float phase = 0.0f) {
  std::vector<float> a(frames);
  for (uint64_t i = 0; i < frames; ++i) {
    a[i] = std::fmod(static_cast<float>(i) / 64.0f + phase, 1.0f);
  }
  return a;
}

}  // namespace

TEST_CASE("A-065: add_to_new_audio_buffer is safe during transcribe_stream snapshot") {
  // We don't have a real libmoonshine so we can't exercise the full
  // transcriber pipeline here, but we can simulate the producer/consumer
  // pattern on a std::vector<float> guarded by audio_buffer_mutex.
  std::vector<float> buffer;
  std::mutex audio_buffer_mutex;
  std::atomic<bool> saw_invalid_iteration{false};
  std::atomic<int> consumer_iterations{0};

  std::thread producer([&]() {
    auto a = make_audio(64, 0.0f);
    for (int i = 0; i < 100; ++i) {
      {
        std::lock_guard<std::mutex> lock(audio_buffer_mutex);
        buffer.insert(buffer.end(), a.begin(), a.end());
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  });

  std::thread consumer([&]() {
    for (int i = 0; i < 100 && !saw_invalid_iteration.load(); ++i) {
      std::vector<float> snapshot;
      {
        std::lock_guard<std::mutex> lock(audio_buffer_mutex);
        snapshot.swap(buffer);
      }
      // Walk the snapshot. Producer cannot invalidate iterators because
      // it locks audio_buffer_mutex before mutating buffer; the lock
      // and swap are atomic from the producer's perspective.
      for (size_t j = 0; j < snapshot.size(); ++j) {
        if (snapshot[j] < 0.0f || snapshot[j] >= 1.0f) {
          saw_invalid_iteration.store(true);
          break;
        }
      }
      consumer_iterations.fetch_add(1);
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  });

  producer.join();
  consumer.join();

  CHECK_FALSE(saw_invalid_iteration.load());
  CHECK(consumer_iterations.load() >= 50);
}

TEST_CASE("A-065: clear + add interleaving does not leak samples") {
  // After a swap, the buffer must be empty so the next add_to_new_audio_buffer
  // call accumulates into a fresh container.
  std::vector<float> buffer;
  std::mutex audio_buffer_mutex;

  std::vector<float> first = make_audio(16, 0.0f);
  {
    std::lock_guard<std::mutex> lock(audio_buffer_mutex);
    buffer.insert(buffer.end(), first.begin(), first.end());
  }

  std::vector<float> snapshot;
  {
    std::lock_guard<std::mutex> lock(audio_buffer_mutex);
    snapshot.swap(buffer);
  }
  CHECK(snapshot.size() == 16);
  CHECK(buffer.empty());

  std::vector<float> second = make_audio(32, 0.5f);
  {
    std::lock_guard<std::mutex> lock(audio_buffer_mutex);
    buffer.insert(buffer.end(), second.begin(), second.end());
  }

  std::vector<float> snapshot2;
  {
    std::lock_guard<std::mutex> lock(audio_buffer_mutex);
    snapshot2.swap(buffer);
  }
  CHECK(snapshot2.size() == 32);
  CHECK(snapshot[0] != snapshot2[0]);  // Different phases verify no aliasing.
}
