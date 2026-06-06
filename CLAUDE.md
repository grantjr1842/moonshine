# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Moonshine Voice — an open-source, on-device, real-time voice AI toolkit. It bundles a C++ core library (`libmoonshine`) with language bindings (Python ctypes, Swift, Java JNI) and pre-built ONNX models. Public app APIs cover transcription, voice activity detection, speaker diarization, intent recognition, and TTS, all event-based and designed for live streaming.

The C++ core depends on Microsoft OnnxRuntime (prebuilt binaries vendored under `core/third-party/onnxruntime/lib/` for all supported platforms).

## Build & test commands

### C++ core (Linux/macOS)

```bash
cd core
mkdir -p build && cd build
cmake ..
cmake --build .
```

Or in one step: `scripts/test-core.sh` (Linux) / `scripts/test-core.bat` (Windows). This rebuilds the core, then runs the entire test suite from the `test-assets/` directory:

- `bin-tokenizer-test` → `core/third-party/onnxruntime/build/onnxruntime-test`
- `moonshine-cpp-test` (built with **C++11** to verify the public header stays backwards-compatible — internal code is C++20)
- `moonshine-c-api-test`, `moonshine-c-api-memory-test`
- `transcriber-test`, `voice-activity-detector-test`, `resampler-test`
- `cosine-distance-test`, `speaker-embedding-model-test`, `online-clusterer-test`
- `word-alignment-test`
- moonshine-tts G2P/rule tests (e.g. `korean_rule_g2p_test`, `english_hand_oov_test`, `text_normalize_test`, `heteronym_context_test`, `onnx_g2p_smoke_test`, `japanese_onnx_g2p_test`, etc.)

`moonshine-c-api-memory-test` runs in a temp dir to verify assets are loaded from memory pointers, not file paths.

`scripts/build-all-platforms.sh` is what runs in CI for releases — must run on macOS, drives remote Linux/Windows GCP VMs for cross-platform builds.

### Python

`scripts/build-pip.sh` — builds the C++ core, copies `libmoonshine.{so,dylib,dll}` + matching `libonnxruntime` into `python/src/moonshine_voice/`, codesigns on macOS, and runs `uv` to build the wheel.

`scripts/build-pip-docker.sh` — same flow inside the provided `Dockerfile` (python:3.12-slim + cmake + uv).

For the published `moonshine-voice` package (PyPI), end users just `pip install moonshine-voice` and use `python -m moonshine_voice.mic_transcriber --language en`.

### Swift / iOS / MacOS

`scripts/build-swift.sh` — produces `swift/Moonshine.xcframework`. CMake builds the core as a **STATIC** `moonshine.framework` (required because iOS cannot embed shared frameworks), then merges `libmoonshine-utils`, `libonnxruntime`, `ort-utils`, `bin-tokenizer`, and the moonshine-tts archives into a single fat static binary via `libtool -static`. Post-build steps patch `Info.plist` (`CFBundleName`, `CFBundleVersion`, `CFBundleShortVersionString`, drop `CFBundleSignature`, convert to binary plist, re-sign).

The Swift package is **separately hosted** at `github.com/moonshine-ai/moonshine-swift` (autoupdated mirror).

### Android

`./gradlew --no-daemon --stacktrace publishAndReleaseToMavenCentral` (from `build.gradle.kts`) — uses signing secrets from env. Version is declared as `coordinates("ai.moonshine", "moonshine-voice", "0.0.62")` in `build.gradle.kts`; CI checks the release tag matches.

`scripts/publish-android.sh` is the script wrapper.

### Examples CI

`scripts/test-examples.sh` downloads each `examples/{ios,android}/<Project>` archive from GitHub Releases, extracts it, and runs the Gradle / xcodebuild build step. `--local-examples` copies in-tree examples instead of downloading.

### C++ formatting

`clang-format.sh` runs `clang-format -i` over all `*.c / *.cc / *.cpp / *.h / *.hpp` files in the tree (skipping `cpp/build` and `.env`). The `.clang-format` file at repo root pins the style. Note: the C++ core is **built with `-Wall -Wextra -pedantic -Werror`** on non-Windows — keep new code warning-clean.

## High-level architecture

### Layered design

```
┌─────────────────────────────────────────────────────────────────┐
│  App examples (examples/{ios,android,macos,python,windows})     │
│  + language bindings (python/src, swift/Sources, android/java,  │
│    android/moonshine-jni)                                       │
├─────────────────────────────────────────────────────────────────┤
│  C++ public C API       (core/moonshine-c-api.{h,cpp})          │
│  C++ public C++ wrapper (core/moonshine-cpp.h, C++11 compatible) │
├─────────────────────────────────────────────────────────────────┤
│  Core pipeline (libmoonshine):                                  │
│    transcriber.cpp → VAD → streaming-model / moonshine-model    │
│                       → speaker-embedding → online-clusterer    │
│                       → word-alignment                          │
│    intent-recognizer.cpp  → gemma-embedding-model               │
│    moonshine-tts/         → g2p + onnx-g2p + kokoro/piper       │
├─────────────────────────────────────────────────────────────────┤
│  Utilities: ort-utils, moonshine-utils, bin-tokenizer,          │
│  third-party/{onnxruntime, doctest, nlohmann, utf-8, utf8proc}   │
└─────────────────────────────────────────────────────────────────┘
```

### Core modules (under `core/`)

The `moonshine` library target (`core/CMakeLists.txt`) compiles ~16 sources together as one shared/static lib:

- **`transcriber.{h,cpp}`** — the high-level pipeline. Owns the `transcriber_t` handle, fans out audio to per-stream VAD+encoder+decoder+speaker-id state, and emits transcript events. Pairs with `voice-activity-detector` (VAD) and `resampler` (audio rate conversion to 16kHz mono).
- **`moonshine-model.{h,cpp}`** — non-streaming encoder/decoder (Tiny, Base).
- **`moonshine-streaming-model.{h,cpp}`** — streaming encoder/decoder with KV cache (Tiny Streaming, Small Streaming, Medium Streaming). This is what gives the low-latency win.
- **`voice-activity-detector.{h,cpp}`** + **`silero-vad.{h,cpp}`** — VAD + the embedded Silero ONNX weights (`silero-vad-model-data.h` is a generated header with the model bytes).
- **`gemma-embedding-model.{h,cpp}`** — sentence-embedding model (Gemma-300M by default) for intent recognition.
- **`intent-recognizer.{h,cpp}`** — wraps the embedding model + cosine distance for semantic phrase matching.
- **`speaker-embedding-model.{h,cpp}`** + **`speaker-embedding-model-data.{h,cpp}`** — extracts speaker embeddings from audio segments.
- **`online-clusterer.{h,cpp}`** — incremental clustering of speaker embeddings (diarization).
- **`word-alignment.{h,cpp}`** — word-level timestamp alignment.
- **`spelling-model.{h,cpp}`** + **`spelling-fusion.{h,cpp}`** + `spelling-fusion-data.{h,cpp}` — alphanumeric spelling mode. `spelling-fusion-data.h` is a generated header with the lookup tables. `MOONSHINE_FLAG_SPELLING_MODE` is set when constructing a transcriber with a `spelling_model_path`.
- **`cosine-distance.{h,cpp}`** — small math helper.
- **`moonshine-c-api.{h,cpp}`** — the C ABI. All exported `moonshine_*` functions. `MOONSHINE_HEADER_VERSION` (currently `20000`) is passed in by callers so newer libs can emulate older ones.
- **`moonshine-cpp.h`** — higher-level C++11 wrapper. Note it's compiled as a separate test with `CXX_STANDARD 11` to enforce backwards compatibility.

### `core/moonshine-tts/` (subtree)

Self-contained TTS + grapheme-to-phoneme (G2P) engine — was previously a separate repo. Owns its own CMake and a `data/` tree of per-language assets (lexicons, OOV ONNX, Kokoro voices, etc.). `moonshine-tts/src/lang-specific/<lang>.cpp` are per-language rule-based G2P modules; `moonshine-tts/src/onnx-g2p.cpp` is the neural fallback. Pulled into `libmoonshine` via `add_subdirectory(moonshine-tts)` and merged at link time (iOS framework / Windows static lib). Toggling is via `MOONSHINE_TTS_BUILD_ONNX`.

### `core/ort-utils/`

Thin wrapper over OnnxRuntime: session creation, tensor allocation, input/output binding. Used by every model in the core.

### `core/moonshine-utils/`

String helpers, debug utilities — no model code, safe to use everywhere.

### Language binding shape

Each binding does the same thing: ctypes / JNI / Swift `import C` over `moonshine-c-api.h`, then a thin idiomatic class on top that emits the event listener callbacks.

- **Python** (`python/src/moonshine_voice/`) — pure ctypes, no C extension. `_load_libc()` then loads the bundled `libmoonshine.{so,dylib,dll}`. Modules: `transcriber.py` (the event-class definitions: `LineStarted`, `LineUpdated`, `LineTextChanged`, `LineCompleted`, plus `Transcriber` class), `mic_transcriber.py` (uses `sounddevice`), `intent_recognizer.py`, `tts.py`, `dialog_flow.py` (multi-step agent), `g2p.py` (grapheme-to-phoneme direct access), `download.py` (fetches model archives into `MOONSHINE_VOICE_CACHE`), `moonshine_api.py` (ctypes boilerplate + raw API mirrors).
- **Swift** (`swift/Sources/MoonshineVoice/`) — calls the C API via `moonshine-c-api.h`. Built as `Moonshine.xcframework` by `scripts/build-swift.sh`.
- **Android** (`android/`) — `moonshine-jni/` is the JNI shim, `java/` is the Kotlin/Java binding.

### Example apps

Each platform has matching example apps that demonstrate the same APIs:

- `examples/python/` — `basic_transcription.py` (WAV file in), `mic_transcription.py` (live mic), `intent_recognition.py`, `dialog_flow.py`, plus the `ollama-voice/` Ollama agent demo and `pocket_tts_verify/`, `stt/`.
- `examples/ios/`, `examples/macos/`, `examples/android/`, `examples/windows/`, `examples/raspberry-pi/`, `examples/c++/` — each is a self-contained project that bundles models under `assets/` (via Git LFS) and depends on the published package (Maven, SPM, NuGet-downloaded lib).

### Model format & lifecycle

Models are OnnxRuntime **flatbuffer `.ort`** files (memory-mappable, faster cold start). `safetensors` versions are on HuggingFace under the `UsefulSensors` org (legacy org name from the company's earlier chip days). `python -m moonshine_voice.download --language <code>` fetches them into `~/Library/Caches/moonshine_voice/download.moonshine.ai/...` (or `$MOONSHINE_VOICE_CACHE` if set). The download script prints the model arch number (`MOONSHINE_MODEL_ARCH_TINY=0`, `BASE=1`, `TINY_STREAMING=2`, `SMALL_STREAMING=4`, `MEDIUM_STREAMING=5`, …) — pass that to `Transcriber(...)`.

Quantization is INT8 weights + INT8 MatMul, with the conv frontend kept at B16 float (B16 is needed for the raw 16-bit audio input to the conv frontend). See `scripts/quantize-streaming-model.sh`.

For non-Latin languages (everything except English and Spanish), set `max_tokens_per_second=13.0` in `Transcriber(..., options={...})` — the default 6.5 is tuned for English and will clip valid output for high-token-rate languages.

## Caching, asset path, and resource locations

- `python/src/moonshine_voice/assets/` ships model files checked into the repo (so the published wheel can be used offline for tiny/base) and `beckett.txt` / `beckett.wav` reference fixtures.
- `test-assets/` is the working dir for `scripts/test-core.sh` — tests `cd` here before running. Models live in `tiny-en/`, `tiny-streaming-en/`, plus `two_cities.wav`, `beckett.wav`, `intent.wav`, `speaker-embedding-model.ort`, `spelling_cnn.ort`, `spelling_cnn_meta.json`.
- `core/third-party/onnxruntime/lib/{linux,macos,windows,ios,android}/` is the vendored prebuilt ORT for every supported platform. The `find-ort-library-path.cmake` module picks the right one.

## Versioning

Version is in three places — **all must stay in sync**:

1. `core/CMakeLists.txt` → `set(MOONSHINE_VERSION "0.0.62")`
2. `python/pyproject.toml` → `version = "0.0.62"`
3. `build.gradle.kts` → `coordinates("ai.moonshine", "moonshine-voice", "0.0.62")`

`scripts/update-version.sh` automates this. CI's `publish-android.yml` validates the release git tag against `build.gradle.kts` and aborts on mismatch.

## Debugging options for the transcriber

Pass via `Transcriber(..., options={...})` (or `--options=...` on the CLI modules):

- `save_input_wav_path=<dir>` — write out exactly what audio the transcriber received as 16kHz mono WAV (`input_1.wav`, `input_2.wav`, …). First thing to check when transcription quality is bad.
- `log_api_calls=true` — trace every C API entry point with args.
- `log_ort_runs=true` — per-ONNX-run timing.
- `log_output_text=true` — print decoded text from STT as it lands.
- `vad_threshold` (default 0.5), `vad_window_duration` (0.5s), `vad_look_behind_sample_count` (8192), `vad_max_segment_duration` (15s, with linearly decreasing threshold after 2/3 of max).
- `max_tokens_per_second` (default 6.5, set to 13.0 for non-Latin languages) — hallucination guard.
- `skip_transcription=true` — VAD/segmentation only; you do downstream processing yourself.
- `transcription_interval` (seconds) — how often the streaming model emits an update. Default 0.5s.
- `identify_speakers` (bool) — runs speaker ID; off by default would save memory, but it's enabled by default.
- `return_audio_data` — whether each `TranscriptLine` carries the raw audio bytes back to the caller.

## Things to be careful about

- **Cross-language identifier conventions**: the C ABI uses snake_case (`moonshine_load_transcriber_from_files`), the C++ wrapper uses `CamelCase`, Python uses `snake_case`, Swift/Kotlin use `CamelCase`. Don't fight it.
- **Memory ownership in the C API**: C strings come back NUL-terminated from `malloc`; Python uses `_decode_utf8_from_c` (in `moonshine_api.py`) which tolerates invalid UTF-8 with `errors="replace"` — don't tighten that.
- **Static vs shared library**: iOS/Swift and Windows static release use a custom `libtool -static` merge step (and Windows uses `lib /OUT:`) to pack `moonshine-utils + ORT + ort-utils + bin-tokenizer + moonshine-tts archives + moonshine` into one file. If you add a new archive dependency, update `core/CMakeLists.txt`'s `_MOONSHINE_FRAMEWORK_MERGE_TTS_ARCHIVES` (iOS) and the parallel Windows `lib /OUT:` block.
- **macOS dylib codesigning**: linker's ad-hoc signature can break `dlopen` from Homebrew Python ("CODESIGNING Invalid Page"). There's a `POST_BUILD` step on macOS shared builds to re-sign with `codesign --force --sign -`. Same fix-up applied to iOS framework's `Info.plist`.
- **`moonshine-cpp-test` is C++11**: this is deliberate to prove the C++ wrapper stays backwards-compatible. Don't use C++20 features in `core/moonshine-cpp.h`.
- **Test cwd matters**: `scripts/test-core.sh` and many C++ tests expect to be run with `test-assets/` as cwd. They use relative paths.
- **Generated headers**: `silero-vad-model-data.h`, `spelling-fusion-data.h`, and `speaker-embedding-model-data.{h,cpp}` contain embedded model bytes. Treat as read-only blobs unless the upstream model is being regenerated.
