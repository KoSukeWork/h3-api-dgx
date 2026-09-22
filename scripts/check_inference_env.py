"""Strict uv check with one verified NVIDIA 0.8.1 ARM wheel-tag exception.

Does not edit wheel metadata, binaries, or dependency versions. Real CUDA execution
is still checked separately by h3_api/preflight.py on the GB10.
"""

import base64
import hashlib
import importlib.metadata
import platform
import re
import subprocess
import sys

PACKAGE = "nvidia-cusparselt-cu13"
LIBRARY = "nvidia/cusparselt/lib/libcusparseLt.so.0"
DIAGNOSTIC = f"The package `{PACKAGE}` was built for a different platform"


def only_known_diagnostic(output):
    """Fail closed on additional incompatibilities, warnings or unknown failures."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if lines.count(DIAGNOSTIC) != 1 or lines.count("Found 1 incompatibility") != 1:
        return False
    return all(
        line in (DIAGNOSTIC, "Found 1 incompatibility")
        or re.fullmatch(r"Using Python .+ environment at: .+", line)
        or re.fullmatch(r"Checked \d+ packages? in .+", line)
        for line in lines
    )


def verify_known_wheel(dist, *, system, machine):
    if system != "Linux" or machine not in ("aarch64", "arm64"):
        raise ValueError("The wheel-tag exception is only valid on Linux ARM64")
    if dist.version != "0.8.1":
        raise ValueError(f"Unaudited cuSPARSELt version: {dist.version}")
    tags = [line for line in (dist.read_text("WHEEL") or "").splitlines() if line.startswith("Tag:")]
    if tags != ["Tag: py3-none-manylinux2014_sbsa"]:
        raise ValueError(f"Unexpected wheel tags: {tags}")
    entries = [entry for entry in (dist.files or []) if str(entry) == LIBRARY]
    if len(entries) != 1:
        raise ValueError("Expected exactly one cuSPARSELt shared library in RECORD")
    entry = entries[0]
    if not entry.hash or entry.hash.mode != "sha256":
        raise ValueError("Missing SHA-256 RECORD entry for the shared library")
    path = dist.locate_file(entry)
    with path.open("rb") as stream:
        header = stream.read(20)
        # ELF64, little endian, EM_AARCH64=183, ET_DYN=3 (shared object).
        if (
            len(header) != 20
            or header[:6] != b"\x7fELF\x02\x01"
            or int.from_bytes(header[16:18], "little") != 3
            or int.from_bytes(header[18:20], "little") != 183
        ):
            raise ValueError("cuSPARSELt library is not an ARM64 ELF shared object")
        stream.seek(0)
        digest = hashlib.file_digest(stream, "sha256").digest()
    actual = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    if actual != entry.hash.value:
        raise ValueError("cuSPARSELt library does not match its RECORD checksum")


def main():
    result = subprocess.run(
        ["uv", "--color", "never", "pip", "check", "--python", sys.executable],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    print(result.stdout, end="", flush=True)
    if result.returncode == 0:
        return
    if result.returncode != 1 or not only_known_diagnostic(result.stdout):
        raise SystemExit(result.returncode)
    try:
        verify_known_wheel(
            importlib.metadata.distribution(PACKAGE),
            system=platform.system(),
            machine=platform.machine(),
        )
    except (ValueError, OSError, importlib.metadata.PackageNotFoundError) as exc:
        raise SystemExit(f"cuSPARSELt exception rejected: {exc}") from exc
    print(
        "Accepted ONLY nvidia-cusparselt-cu13==0.8.1's known sbsa tag mismatch: "
        "Linux ARM64, expected WHEEL tag, AArch64 ELF and RECORD checksum verified. "
        "Package contents unchanged; GPU acceptance is still required."
    )


if __name__ == "__main__":
    main()
