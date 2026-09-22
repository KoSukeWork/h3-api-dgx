"""Download only selected, pinned benchmark assets; verify every byte with SHA256."""

import argparse
import hashlib
import json
from pathlib import Path

ASSETS_ROOT = Path("/srv/h3/data/benchmark/assets")
ASSETS = {
    "turbo": {
        "repo": "larryvrh/MiniMax-H3-Turbo-Lora",
        "revision": "43a74557ac3f6539db8e0f2a959d03feb7a81480",
        "file": "minimax_h3_turbo_v4_step600_ema.safetensors",
        "bytes": 779849816,
        "sha256": "5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3",
    },
    "int8": {
        "repo": "Comfy-Org/MiniMax-H3",
        "revision": "7e75982b97cd5a41d2dcfa1904ee88d0686d6fd1",
        "file": "diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors",
        "bytes": 34038892334,
        "sha256": "7ad4c73e6e378b822ffd1629f27f632d3787d95f5e468e3af958f98c58df96a5",
    },
}


def verify(mode, root=ASSETS_ROOT):
    spec = ASSETS[mode]
    path = root / mode / spec["file"]
    if path.stat().st_size != spec["bytes"]:
        raise ValueError(f"Wrong asset size: {path}")
    print(f"Verifying {mode} SHA256 (outside generation timing)...", flush=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != spec["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=ASSETS)
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download

    spec = ASSETS[args.mode]
    print(f"Download {spec['repo']}/{spec['file']} ({spec['bytes'] / 1e9:.2f} GB)", flush=True)
    hf_hub_download(
        repo_id=spec["repo"],
        filename=spec["file"],
        revision=spec["revision"],
        local_dir=ASSETS_ROOT / args.mode,
    )
    path = verify(args.mode)
    print(json.dumps({"path": str(path), **spec}, indent=2), flush=True)


if __name__ == "__main__":
    main()
