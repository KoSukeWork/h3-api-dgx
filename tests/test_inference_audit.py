"""Synthetic metadata tests only: no torch, CUDA, wheels or GPU execution."""

import base64
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "inference_audit", Path(__file__).parents[1] / "scripts" / "check_inference_env.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fake_dist(tmp_path, *, version="0.8.1", machine=183, tag="manylinux2014_sbsa", corrupt=False):
    data = bytearray(64)
    data[:6] = b"\x7fELF\x02\x01"
    data[16:18] = (3).to_bytes(2, "little")
    data[18:20] = machine.to_bytes(2, "little")
    path = tmp_path / "synthetic.so"
    path.write_bytes(data)

    class Entry(str):
        hash = SimpleNamespace(
            mode="sha256",
            value="bad"
            if corrupt
            else base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode(),
        )

    return SimpleNamespace(
        version=version,
        files=[Entry(audit.LIBRARY)],
        read_text=lambda _: f"Wheel-Version: 1.0\nTag: py3-none-{tag}\n",
        locate_file=lambda _: path,
    )


def test_accept_only_single_exact_uv_diagnostic():
    message = "Using Python 3.12.3 environment at: /opt/venv\nChecked 232 packages in 3ms\n"
    message += "Found 1 incompatibility\n" + audit.DIAGNOSTIC + "\n"
    assert audit.only_known_diagnostic(message)
    assert not audit.only_known_diagnostic(message.replace("1 incompatibility", "2 incompatibilities"))
    assert not audit.only_known_diagnostic(message + "Another package is broken\n")
    assert not audit.only_known_diagnostic(message.replace("nvidia-cusparselt-cu13", "torch"))


def test_accept_known_arm_metadata_and_elf(tmp_path):
    audit.verify_known_wheel(fake_dist(tmp_path), system="Linux", machine="aarch64")


@pytest.mark.parametrize(
    "kwargs",
    [{"version": "0.9.1"}, {"machine": 62}, {"tag": "manylinux2014_x86_64"}, {"corrupt": True}],
)
def test_reject_other_version_architecture_tag_or_corruption(tmp_path, kwargs):
    with pytest.raises(ValueError):
        audit.verify_known_wheel(fake_dist(tmp_path, **kwargs), system="Linux", machine="aarch64")


def test_reject_x86_host(tmp_path):
    with pytest.raises(ValueError, match="only valid on Linux ARM64"):
        audit.verify_known_wheel(fake_dist(tmp_path), system="Linux", machine="x86_64")
