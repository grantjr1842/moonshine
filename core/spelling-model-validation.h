#ifndef SPELLING_MODEL_VALIDATION_H
#define SPELLING_MODEL_VALIDATION_H

#include <charconv>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace spelling_model_validation {

constexpr int32_t kMinSampleRate = 8000;
constexpr int32_t kMaxSampleRate = 48000;
constexpr float kMaxClipSeconds = 30.0f;
constexpr size_t kMaxTargetSamples = 1440000;
constexpr size_t kMaxClasses = 256;
constexpr size_t kMaxClassLength = 128;

inline std::string_view trim(std::string_view value) {
  while (!value.empty() && value.front() <= ' ') value.remove_prefix(1);
  while (!value.empty() && value.back() <= ' ') value.remove_suffix(1);
  return value;
}

inline bool parse_sample_rate(std::string_view raw, int32_t *out) {
  if (out == nullptr) return false;
  raw = trim(raw);
  if (raw.empty()) return false;
  int64_t value = 0;
  const char *begin = raw.data();
  const char *end = begin + raw.size();
  auto result = std::from_chars(begin, end, value);
  if (result.ec != std::errc() || result.ptr != end ||
      value < kMinSampleRate || value > kMaxSampleRate) {
    return false;
  }
  *out = static_cast<int32_t>(value);
  return true;
}

inline bool parse_clip_seconds(std::string_view raw, float *out) {
  if (out == nullptr) return false;
  raw = trim(raw);
  if (raw.empty()) return false;
  std::string owned(raw);
  char *end = nullptr;
  errno = 0;
  float value = std::strtof(owned.c_str(), &end);
  if (errno == ERANGE || end != owned.c_str() + owned.size() ||
      !std::isfinite(value) || value <= 0.0f || value > kMaxClipSeconds) {
    return false;
  }
  *out = value;
  return true;
}

inline bool compute_target_samples(int32_t sample_rate, float clip_seconds,
                                  size_t *out) {
  if (out == nullptr || sample_rate < kMinSampleRate ||
      sample_rate > kMaxSampleRate || !std::isfinite(clip_seconds) ||
      clip_seconds <= 0.0f || clip_seconds > kMaxClipSeconds) {
    return false;
  }
  long double product = static_cast<long double>(sample_rate) * clip_seconds;
  if (!std::isfinite(product) || product <= 0.0L ||
      product > static_cast<long double>(kMaxTargetSamples)) {
    return false;
  }
  size_t samples = static_cast<size_t>(std::llround(product));
  if (samples == 0 || samples > kMaxTargetSamples) return false;
  *out = samples;
  return true;
}

inline bool validate_classes(const std::vector<std::string> &classes) {
  if (classes.empty() || classes.size() > kMaxClasses) return false;
  for (const auto &label : classes) {
    if (label.empty() || label.size() > kMaxClassLength) return false;
  }
  return true;
}

}  // namespace spelling_model_validation

#endif  // SPELLING_MODEL_VALIDATION_H
