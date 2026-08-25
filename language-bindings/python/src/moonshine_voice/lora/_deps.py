"""Optional-extra checks for the LoRA training path.

This module is stdlib-only so ``import moonshine_voice.lora`` stays cheap for
inference installs. The heavy packages (PyTorch, Transformers, …) are listed
here and imported only after ``require_lora_deps()`` succeeds.
"""

from __future__ import annotations

# Packages needed to prepare data and train an adapter. Keep the union of
# these tables in sync with ``[project.optional-dependencies] lora`` in
# pyproject.toml. Evaluation and export are optional operations, so they are
# checked only when requested instead of making a basic training run install
# ONNX or jiwer.
TRAINING_PACKAGES = (
    ("torch", "torch"),
    ("transformers", "transformers>=5.15"),
    ("accelerate", "accelerate>=1.0"),
    ("OpenSSL", "pyOpenSSL>=24.2.1"),
    ("safetensors", "safetensors"),
    ("soundfile", "soundfile"),
    ("pyarrow", "pyarrow"),
    ("scipy", "scipy"),
    ("huggingface_hub", "huggingface_hub"),
)

INSTALL_HINT = "pip install 'moonshine-voice[finetune]'"
INSTALL_HINT_ALIASES = "pip install 'moonshine-voice[finetune]'  (or 'moonshine-voice[lora]')"
EVAL_PACKAGES = (("jiwer", "jiwer"),)
EXPORT_PACKAGES = (("onnx", "onnx"), ("onnxscript", "onnxscript"))

# Public compatibility name for callers that used the original helper.
LORA_PACKAGES = TRAINING_PACKAGES + EVAL_PACKAGES + EXPORT_PACKAGES


def missing_lora_packages(*, include_eval=False, include_export=False):
    """Return pip specifiers for extra packages that are not importable."""
    packages = list(TRAINING_PACKAGES)
    if include_eval:
        packages.extend(EVAL_PACKAGES)
    if include_export:
        packages.extend(EXPORT_PACKAGES)
    missing = []
    for module, spec in packages:
        try:
            if module == "transformers":
                # A top-level import can succeed while the model class still
                # fails because an optional backend is broken or incomplete.
                from transformers import (  # noqa: F401
                    AutoProcessor,
                    MoonshineStreamingForConditionalGeneration,
                )
            else:
                __import__(module)
        except (ImportError, OSError, RuntimeError, AttributeError):
            missing.append(spec)
    return missing


def lora_deps_error(missing):
    names = ", ".join(missing)
    return ImportError(
        "Fine-tuning needs extra packages that are not installed with the "
        "default moonshine-voice inference wheel.\n"
        f"  {INSTALL_HINT_ALIASES}\n"
        f"Missing: {names}"
    )


def require_lora_deps(*, include_eval=False, include_export=False):
    """Raise ``ImportError`` with an install hint if the extra is missing."""
    missing = missing_lora_packages(
        include_eval=include_eval, include_export=include_export
    )
    if missing:
        raise lora_deps_error(missing)
