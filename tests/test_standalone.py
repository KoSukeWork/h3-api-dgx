import hashlib
from pathlib import Path

from scripts.download_models import assets, matches

ROOT = Path(__file__).resolve().parent.parent


def test_download_manifest_is_complete_and_nonquantized():
    selected = list(assets())
    assert len(selected) == 6
    assert all("int8" not in item[2] and "fp8" not in item[2] for item in selected)
    assert sum(item[4] for item in selected) == 190660183112
    assert all(len(item[1]) == 40 and len(item[5]) == 64 for item in selected)


def test_existing_models_must_match_size_and_hash(tmp_path):
    path = tmp_path / "model"
    path.write_bytes(b"abc")
    assert matches(path, 3, hashlib.sha256(b"abc").hexdigest())
    assert not matches(path, 4, hashlib.sha256(b"abc").hexdigest())
    assert not matches(path, 3, hashlib.sha256(b"xyz").hexdigest())


def test_constraints_are_derived_from_actual_environment():
    source = (ROOT / "docs/inference-resolved-20260922.txt").read_text().splitlines()
    pins = (ROOT / "inference-constraints.txt").read_text().splitlines()
    assert {s for s in source if "==" in s} == {s for s in pins if not s.startswith("#")}
    assert "torch==2.13.0+cu130" in pins
    assert not any(s.startswith("-e") for s in pins)


def test_ca_precedes_external_installs_and_both_patches_are_available():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert dockerfile.index("bash /tmp/install_ca.sh") < dockerfile.index("pip install")
    assert dockerfile.index("bash /tmp/install_ca.sh") < dockerfile.index("git clone")
    assert "--constraint /tmp/inference-constraints.txt" in dockerfile
    assert (ROOT / "scripts/patch_h3_qkv.py").is_file()
    assert (ROOT / "benchmark/patch_lora_compat.py").is_file()
