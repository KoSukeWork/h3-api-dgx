import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from h3_api import prepare
from h3_api.backend import SGLangBackend
from h3_api.settings import Settings
from scripts import audit_h3_qkv_layout as audit
from scripts import patch_h3_qkv as patch


def test_row_mapping_is_a_permutation_and_not_identity():
    rows = [audit.grouped_row(i) for i in range(21504)]
    assert sorted(rows) == list(range(21504))
    assert [rows[i] for i in (128, 7168, 7296, 14336)] == [384, 128, 512, 256]


@pytest.mark.parametrize("flag,quantized,diffusers,expected", [
    (None, False, False, False),
    ("concatenated", False, False, True),
    ("concatenated", True, False, False),
    (None, False, True, True),
])
def test_patch_only_bypasses_reorder_for_explicit_unquantized_layout(
    monkeypatch, flag, quantized, diffusers, expected
):
    # Exercise the actual replacement branch without torch, CUDA or SGLang imports.
    source = (
        "def select(arch, quant_config):\n"
        "    if True:\n"
        "        checkpoint_qkv_is_native = False\n"
        + patch.ORIGINAL
        + "        return checkpoint_qkv_is_native\n"
    )
    monkeypatch.setattr(patch, "SOURCE_SHA256", hashlib.sha256(source.encode()).hexdigest())
    updated = patch.patched_source(source)
    assert patch.patched_source(updated) == updated
    env = {} if flag is None else {"H3_COMFY_BF16_QKV_LAYOUT": flag}
    scope = {"os": SimpleNamespace(environ=env)}
    exec(updated, scope)  # noqa: S102 - Execute only the fixed, locally authored test fixture.
    assert scope["select"](
        SimpleNamespace(checkpoint_uses_diffusers_layout=diffusers), object() if quantized else None
    ) is expected


def test_patch_refuses_unknown_source():
    with pytest.raises(ValueError, match="another revision"):
        patch.patched_source("# unrecognized upstream version\n")


@pytest.mark.parametrize("variant", ["fl2va", "ref2va"])
def test_gateway_contract_and_fresh_spill_namespace(tmp_path, monkeypatch, variant):
    settings = Settings(tmp_path / "data", tmp_path / "models", "test-key")
    root = settings.models_dir / "prepared"
    root.mkdir(parents=True)
    weight = settings.models_dir / "example.safetensors"
    weight.write_bytes(b"tiny synthetic fixture")
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    size = weight.stat().st_size
    monkeypatch.setitem(prepare.WEIGHTS, variant, (weight.name, size, digest, {"BF16"}))
    manifest = {
        "weights_revision": prepare.WEIGHTS_REVISION,
        "variants": {variant: {"transformer": str(weight)}},
        "verified_weights": {variant: {"sha256": digest, "bytes": size, "path": str(weight)}},
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    backend = SGLangBackend(settings)
    try:
        monkeypatch.setenv("H3_COMFY_BF16_QKV_LAYOUT", "untrusted-inherited-value")
        assert "H3_COMFY_BF16_QKV_LAYOUT" not in backend.launch_context()["env"]
        env = backend.launch_context(variant)["env"]
        assert env["H3_COMFY_BF16_QKV_LAYOUT"] == "concatenated"
        assert Path(env["SGLANG_DIFFUSION_HOST_SPILL_DIR"]) == (
            settings.data_dir / "cache" / "host_spill" / "comfy-bf16-concat-v1"
        )
        manifest["verified_weights"][variant]["sha256"] = "unknown-checkpoint"
        path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="SHA-verified"):
            backend.launch_context(variant)
        manifest["verified_weights"][variant]["sha256"] = digest
        path.write_text(json.dumps(manifest))
        weight.write_bytes(b"replaced file")
        with pytest.raises(ValueError, match="SHA-verified"):
            backend.launch_context(variant)
    finally:
        asyncio.run(backend.close())


def test_audit_rejects_server_ignoring_range(monkeypatch):
    class Response:
        status = 200

        def __init__(self):
            self.headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, size):
            pytest.fail("Must not start downloading a whole shard")

    monkeypatch.setattr(audit.urllib.request, "urlopen", lambda *a, **kw: Response())
    with pytest.raises(RuntimeError, match="whole-shard"):
        audit.remote_range("https://example.invalid/weights", 0, 8)
