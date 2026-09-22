import hashlib
from types import SimpleNamespace

import pytest

from benchmark import patch_lora_compat as patch


def capability_function():
    scope = {"nn": SimpleNamespace(Module=object)}
    exec(patch.REPLACEMENT, scope)  # noqa: S102 - Only locally authored helper, no GPU code.
    return scope["_accepts_mxfp8_input"]


def test_lora_wrapper_without_quant_method_uses_tensor_path():
    class BaseMethod:
        def accepts_mxfp8_input(self, layer):
            pytest.fail("Must not unwrap LoRA and select a packed input for its base")

    wrapper = SimpleNamespace(base_layer=SimpleNamespace(quant_method=BaseMethod()))
    with pytest.raises(AttributeError):
        # Reproduce the old capability probe failure with a CPU-only object.
        _ = wrapper.quant_method
    assert capability_function()(wrapper) is False


@pytest.mark.parametrize("accepts", [True, False])
def test_native_quant_method_receives_original_layer(accepts):
    calls = []
    method = SimpleNamespace(accepts_mxfp8_input=lambda layer: calls.append(layer) or accepts)
    layer = SimpleNamespace(quant_method=method)
    assert capability_function()(layer) is accepts
    assert calls == [layer]


def test_none_method_still_returns_false():
    assert capability_function()(SimpleNamespace(quant_method=None)) is False


def test_patch_is_narrow_idempotent_and_protects_existing_qkv_fix(monkeypatch):
    source = "# Existing QKV compatibility remains untouched\n" + patch.ORIGINAL
    monkeypatch.setattr(patch, "SOURCE_SHA256", hashlib.sha256(source.encode()).hexdigest())
    result = patch.patched_source(source)
    assert result.replace(patch.REPLACEMENT, patch.ORIGINAL) == source
    assert patch.patched_source(result) == result
    with pytest.raises(ValueError, match="Unexpected"):
        patch.patched_source(result + "# changed elsewhere\n")


def test_unknown_version_refused():
    with pytest.raises(ValueError, match="Unexpected"):
        patch.patched_source(patch.ORIGINAL)
