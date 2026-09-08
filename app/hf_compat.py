"""Use CUDA torchvision when it matches torch; skip it when it does not.

A CUDA torch wheel without a matching torchvision build can raise
``RuntimeError: operator torchvision::nms does not exist`` or hard-crash
Windows (0xc0000139). Transformers imports torchvision while loading
XLM-RoBERTa (BGE reranker). If the CUDA wheels match, leave it enabled.
"""

from __future__ import annotations

import sys

_patched = False


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
