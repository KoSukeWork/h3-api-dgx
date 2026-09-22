"""Download the five pinned BF16-baseline components and Turbo adapter, never INT8."""

import hashlib
import platform
from pathlib import Path

from h3_api.prepare import WEIGHTS, WEIGHTS_REVISION
from h3_api.turbo import LORA_BYTES, LORA_FILENAME, LORA_SHA256

MODELS = Path("/srv/h3/models")


def assets():
    for filename, size, digest, _ in WEIGHTS.values():
        yield "Comfy-Org/MiniMax-H3", WEIGHTS_REVISION, filename, MODELS, size, digest
    yield (
        "larryvrh/MiniMax-H3-Turbo-Lora",
        "43a74557ac3f6539db8e0f2a959d03feb7a81480",
        LORA_FILENAME,
        MODELS / "loras",
        LORA_BYTES,
        LORA_SHA256,
    )


def matches(path, size, digest):
    if not path.is_file() or path.stat().st_size != size:
        return False
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    return actual == digest


def main():
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("Run this setup download in the ARM64 DGX container")
    from huggingface_hub import hf_hub_download

    for repo, revision, filename, root, size, digest in assets():
        path = root / filename
        print(f"Check {filename} ({size / 1e9:.2f} GB)", flush=True)
        if path.exists():
            if not matches(path, size, digest):
                raise ValueError(f"Existing file differs from pinned SHA256; refusing to overwrite {path}")
            print("Already verified; skipping download.", flush=True)
            continue
        hf_hub_download(repo_id=repo, filename=filename, revision=revision, local_dir=root)
        if not matches(path, size, digest):
            raise ValueError(f"SHA256 mismatch: {path}")
        path.chmod(0o644)
        print("SHA256 verified.", flush=True)


if __name__ == "__main__":
    main()
