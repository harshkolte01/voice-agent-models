from __future__ import annotations

from unittest.mock import patch

import transformers.utils.import_utils as import_utils

from app import hf_compat


def test_disable_broken_torchvision_keeps_matching_cuda_wheel() -> None:
    hf_compat._patched = False
    import_utils._torchvision_available = True
    hf_compat.disable_broken_torchvision()
    assert import_utils._torchvision_available is True
    hf_compat._patched = False


def test_disable_broken_torchvision_skips_nms_mismatch() -> None:
    hf_compat._patched = False
    import_utils._torchvision_available = True

    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torchvision" or name.startswith("torchvision."):
            raise RuntimeError("operator torchvision::nms does not exist")
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=fake_import):
        hf_compat.disable_broken_torchvision()

    assert import_utils._torchvision_available is False
    import_utils._torchvision_available = True
    hf_compat._patched = False
