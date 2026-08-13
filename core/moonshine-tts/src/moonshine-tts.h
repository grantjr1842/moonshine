#ifndef MOONSHINE_TTS_MOONSHINE_TTS_H
#define MOONSHINE_TTS_MOONSHINE_TTS_H

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "moonshine-g2p-options.h"
#include "moonshine-tts-options.h"

namespace moonshine_tts {

bool kokoro_tts_lang_supported(std::string_view lang_cli,
                               const MoonshineG2POptions& g2p_opt = {});

/// Unified TTS: **Kokoro** and **Piper** ONNX backends; shared ``MoonshineG2P``
/// where applicable.
class MoonshineTTS {
 public:
  MoonshineTTS(std::string_view language, const MoonshineTTSOptions& opt);
  MoonshineTTS(const MoonshineTTS&) = delete;
  MoonshineTTS& operator=(const MoonshineTTS&) = delete;
  MoonshineTTS(MoonshineTTS&&) noexcept;
  MoonshineTTS& operator=(MoonshineTTS&&) noexcept;
  ~MoonshineTTS();

  static constexpr int kSampleRateHz = 24000;

  std::vector<float> synthesize(std::string_view text);

  /// Synthesize with per-call option overrides (same key names as
  /// ``MoonshineTTSOptions::parse_options``). Recognized keys: ``speed``,
  /// ``normalize_audio`` (legacy alias ``piper_normalize_audio``),
  /// ``output_volume`` (legacy alias ``piper_output_volume``). Other entries
  /// are ignored.
  std::vector<float> synthesize(
      std::string_view text,
      const std::vector<std::pair<std::string, std::string>>& option_overrides);

  /// Synthesize from an existing IPA phoneme string, skipping grapheme-to-
  /// phoneme conversion. ``phonemes`` is the same International Phonetic
  /// Alphabet representation produced by ``MoonshineG2P::text_to_ipa`` (i.e.
  /// the ``moonshine_text_to_phonemes`` C API), so callers can inspect or edit
  /// the phonemes between G2P and vocoding. The string is normalized to the
  /// active engine's phoneme inventory before synthesis.
  std::vector<float> synthesize_from_phonemes(std::string_view phonemes);

  /// ``synthesize_from_phonemes`` with per-call option overrides (same keys as
  /// the ``synthesize`` overload: ``speed``, ``normalize_audio``,
  /// ``output_volume``).
  std::vector<float> synthesize_from_phonemes(
      std::string_view phonemes,
      const std::vector<std::pair<std::string, std::string>>& option_overrides);

  /// Per-chunk callback used by the streaming TTS path. Returns ``true`` to
  /// continue receiving chunks, ``false`` to abort synthesis. ``samples`` /
  /// ``sample_count`` are valid for the duration of the callback only.
  /// ``sample_rate_hz`` is the engine sample rate (24 000 Hz for Kokoro /
  /// Piper / ZipVoice). ``is_final`` is true for the last chunk; earlier
  /// chunks have ``is_final == false``. ``user_data`` is the pointer the
  /// caller passed to ``synthesize_stream``.
  using ChunkCallback = bool (*)(const float* samples, uint64_t sample_count,
                                int32_t sample_rate_hz, bool is_final,
                                void* user_data);

  /// Streaming TTS: synthesize ``text`` and emit one callback per phoneme
  /// chunk (the existing internal chunked pipeline). Returns
  /// ``MOONSHINE_ERROR_NONE`` if every chunk was emitted successfully, the
  /// chunk callback's ``false`` return value (negated: ``1``) if a callback
  /// aborted, or one of the engine's error codes on failure. The final
  /// chunk's callback is invoked with ``is_final == true`` before this method
  /// returns.
  int32_t synthesize_stream(std::string_view text, ChunkCallback on_chunk,
                            void* user_data);

  /// ``synthesize_stream`` with per-call option overrides (same keys as
  /// ``synthesize``: ``speed``, ``normalize_audio``, ``output_volume``).
  int32_t synthesize_stream(
      std::string_view text, ChunkCallback on_chunk, void* user_data,
      const std::vector<std::pair<std::string, std::string>>& option_overrides);

  /// Returns true if the active engine supports streaming synthesis (all
  /// bundled backends do; useful for capability checks before calling
  /// ``synthesize_stream``).
  bool supports_streaming() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

void write_wav_mono_pcm16(const std::filesystem::path& path,
                          const std::vector<float>& samples);

/// Kokoro or Piper vocoder asset keys only (no G2P), relative to ``g2p_root``.
/// With default ``MoonshineTTSOptions{}``, uses ``vocoder_engine=auto`` and
/// default voice layout.
std::vector<std::string> moonshine_catalog_tts_vocoder_only_dependency_keys(
    std::string_view language_cli);
/// Uses ``voice`` (optional ``kokoro_`` / ``piper_`` prefix sets vocoder),
/// Piper/Kokoro file map entries, and
/// ``g2p_options`` (e.g. Spanish narrow obstruents for auto engine) like
/// ``MoonshineTTS``.
std::vector<std::string> moonshine_catalog_tts_vocoder_only_dependency_keys(
    std::string_view language_cli, const MoonshineTTSOptions& options);

/// Union of vocoder keys across all languages in
/// ``moonshine_asset_catalog_all_registered_language_tags``.
std::vector<std::string>
moonshine_catalog_all_tts_vocoder_dependency_keys_union();

/// One Kokoro or Piper voice id and whether the asset is available (on disk or
/// in-memory file map).
struct MoonshineTtsVoiceAvailability {
  std::string id;
  bool available = false;
};

/// All known voices for ``language_cli`` with availability, using the same path
/// layout rules as
/// ``moonshine_catalog_tts_vocoder_only_dependency_keys``. Returned ``id``
/// values are prefixed with ``kokoro_`` or
/// ``piper_``. When vocoder is ``auto``, Kokoro and Piper catalogs are merged
/// (both prefixes). Kokoro uses the upstream Kokoro-82M voice catalog
/// (VOICES.md) plus any extra ``*.kokorovoice`` under the resolved voices
/// directory. Piper uses the language default ONNX stem plus any ``*.onnx`` in
/// the resolved voices directory. The ``voice`` field in ``options`` does not
/// filter the list.
std::vector<MoonshineTtsVoiceAvailability>
moonshine_list_tts_voices_with_availability(std::string_view language_cli,
                                            const MoonshineTTSOptions& options);

}  // namespace moonshine_tts

#endif
