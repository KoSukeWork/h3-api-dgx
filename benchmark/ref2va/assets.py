"""Pinned original BF16 PEFT adapter; full SHA verification before each test."""

import hashlib
import json
from pathlib import Path

ROOT = Path("/srv/h3/ref2va-test/assets")
SPEC = {
    "repo": "lightx2v/Minimax-h3-Turbo",
    "revision": "30f485ed874aa44c8183348ad361c5eabad318da",
    "file": "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors",
    "bytes": 1383677768,
    "sha256": "9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99",
}


def verify(root=ROOT):
    path = root / SPEC["file"]
    if path.stat().st_size != SPEC["bytes"]:
        raise ValueError("Ref2VA LoRA file size mismatch")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != SPEC["sha256"]:
        raise ValueError("Ref2VA LoRA SHA256 mismatch")
    return path


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    hf_hub_download(repo_id=SPEC["repo"], revision=SPEC["revision"],
                    filename=SPEC["file"], local_dir=ROOT)
    print(json.dumps({"verified_path": str(verify()), **SPEC}, indent=2), flush=True)
