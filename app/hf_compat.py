"""Use CUDA torchvision when it matches torch; skip it when it does not.

A CUDA torch wheel without a matching torchvision build can raise
``RuntimeError: operator torchvision::nms does not exist`` or hard-crash
Windows (0xc0000139). Transformers imports torchvision while loading
XLM-RoBERTa (BGE reranker). If the CUDA wheels match, leave it enabled.

Also force Hugging Face Hub to copy cache files on Windows. Concurrent
``snapshot_download`` threads can race the symlink probe and raise
``OSError: [WinError 1314]`` even after the “symlinks not supported” warning.
"""

from __future__ import annotations

import os
import sys

_patched = False
_hf_symlinks_patched = False


def disable_broken_torchvision() -> None:
    global _patched
    if _patched:
        return

    try:
        from torchvision.transforms import InterpolationMode  # noqa: F401
    except Exception:
        for name in list(sys.modules):
            if name == "torchvision" or name.startswith("torchvision."):
                del sys.modules[name]
        import transformers.utils.import_utils as import_utils

        import_utils._torchvision_available = False

    _patched = True


def force_hf_hub_copy_cache() -> None:
    """Disable HF Hub symlinks on Windows so model downloads cannot WinError 1314."""
    global _hf_symlinks_patched
    if _hf_symlinks_patched or os.name != "nt":
        return

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        import huggingface_hub.file_download as file_download
    except ImportError:
        return

    def _no_symlinks(cache_dir=None) -> bool:  # noqa: ARG001
        return False

    file_download.are_symlinks_supported = _no_symlinks  # type: ignore[assignment]
    supported = getattr(file_download, "_are_symlinks_supported_in_dir", None)
    if isinstance(supported, dict):
        for key in list(supported):
            supported[key] = False

    _hf_symlinks_patched = True
