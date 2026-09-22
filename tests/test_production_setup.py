import inspect
from pathlib import Path

import pytest

from production import manage


def test_env_is_secret_and_never_overwritten(tmp_path):
    path = tmp_path / ".env"
    manage.write_env(path)
    original = path.read_bytes()
    assert b"H3_BIND_IP=127.0.0.1" in original
    assert len(original.splitlines()[0].split(b"=", 1)[1]) >= 32
    manage.write_env(path)
    assert path.read_bytes() == original


def test_lora_copy_retains_source_and_never_overwrites(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "dest"
    source.write_bytes(b"verified-lora")

    def verify(path):
        if path.read_bytes() != b"verified-lora":
            raise ValueError("bad hash")

    monkeypatch.setattr(manage, "verify_lora", verify)
    manage.install_lora(source, destination)
    assert source.read_bytes() == destination.read_bytes() == b"verified-lora"
    destination.write_bytes(b"user-file")
    with pytest.raises(ValueError):
        manage.install_lora(source, destination)
    assert destination.read_bytes() == b"user-file"


def test_preparation_rejects_x86(monkeypatch):
    monkeypatch.setattr(manage.platform, "machine", lambda: "AMD64")
    with pytest.raises(RuntimeError, match="ARM64"):
        manage.host_guard()


def test_interrupted_lora_copy_leaves_original_and_no_partial_model(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "dest"
    source.write_bytes(b"original")
    monkeypatch.setattr(manage, "verify_lora", lambda path: None)

    def broken_copy(origin, target, length):
        target.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(manage.shutil, "copyfileobj", broken_copy)
    with pytest.raises(OSError, match="disk full"):
        manage.install_lora(source, destination)
    assert source.read_bytes() == b"original"
    assert not destination.exists()
    assert not list(tmp_path.glob(".h3-lora-*"))


def test_cleanup_inventory_is_read_only():
    source = inspect.getsource(manage.inventory)
    assert "rmtree" not in source
    assert ".unlink(" not in source
    assert "shutil.move" not in source


def test_inventory_uses_current_package_and_optional_explicit_legacy_path(tmp_path):
    paths = manage.inventory_paths()
    assert manage.PACKAGE in paths
    assert manage.MODELS in paths and manage.DATA in paths
    legacy = tmp_path / "old-install"
    assert legacy not in paths
    assert legacy in manage.inventory_paths(legacy)
    assert manage.inventory_paths(manage.PACKAGE).count(manage.PACKAGE) == 1
    with pytest.raises(ValueError, match="absolute"):
        manage.inventory_paths(Path("relative-install"))
