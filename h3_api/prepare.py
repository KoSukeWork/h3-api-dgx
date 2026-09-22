"""Verify the downloaded weights and prepare a small native SGLang config tree."""

import argparse
import hashlib
import json
import math
import struct
import time
from pathlib import Path

OFFICIAL_REPO = "MiniMaxAI/MiniMax-H3"
OFFICIAL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
WEIGHTS_REVISION = "7e75982b97cd5a41d2dcfa1904ee88d0686d6fd1"
WEIGHTS = {
    "fl2va": (
        "diffusion_models/minimax_h3_fl2va_bf16.safetensors",
        66280487368,
        "907d4add438438ec1544f5240c3b38532ed934fe6be75677a6bbda2a6fdd6182",
        {"BF16", "F32"},
    ),
    "ref2va": (
        "diffusion_models/minimax_h3_ref2va_bf16.safetensors",
        66280487368,
        "e32c54c1a7b4f5f397f195cea267ccb18806303bb665678c4bee60953bdf3026",
        {"BF16", "F32"},
    ),
    "text_encoder": (
        "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors",
        51506295256,
        "600d567f6a9629c8574e8e7041b199bdd9c59a986afa7906910a81919610607d",
        {"BF16"},
    ),
    "video_vae": (
        "vae/minimax_h3_video_vae_fp16.safetensors",
        5207808496,
        "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
        {"F16"},
    ),
    "audio_vae": (
        "vae/minimax_h3_audio_vae_fp32.safetensors",
        605254808,
        "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
        {"F32"},
    ),
}


def inspect_header(path: Path, allowed_dtypes: set[str]) -> dict:
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"Truncated safetensors file: {path}")
        length = struct.unpack("<Q", length_bytes)[0]
        if not 2 <= length <= 100 * 1024 * 1024:
            raise ValueError(f"Invalid header length: {path}")
        tensors = json.loads(stream.read(length))
    tensors.pop("__metadata__", None)
    if not tensors:
        raise ValueError(f"Empty checkpoint: {path}")
    spans = []
    dtypes = set()
    for name, tensor in tensors.items():
        dtype = tensor["dtype"]
        if dtype not in allowed_dtypes or name.endswith(".comfy_quant"):
            raise ValueError(f"Unexpected precision/quantization in {path}: {name} {dtype}")
        dtypes.add(dtype)
        start, end = tensor["data_offsets"]
        expected = math.prod(tensor["shape"]) * {"BF16": 2, "F16": 2, "F32": 4}[dtype]
        if start < 0 or end - start != expected:
            raise ValueError(f"Invalid tensor size: {path}: {name}")
        spans.append((start, end))
    cursor = 0
    for start, end in sorted(spans):
        if start != cursor:
            raise ValueError(f"Overlapping/non-contiguous tensor data: {path}")
        cursor = end
    if cursor != path.stat().st_size - 8 - length:
        raise ValueError(f"Incomplete checkpoint data: {path}")
    return {"tensor_count": len(tensors), "dtypes": sorted(dtypes)}


def verify(models: Path, *, hashes=True):
    report = {}
    for name, (relative, size, expected, dtypes) in WEIGHTS.items():
        path = models / relative
        if path.stat().st_size != size:
            raise ValueError(f"Wrong file size: {path}; expected {size}")
        entry = inspect_header(path, dtypes)
        if hashes:
            print(f"SHA-256: {relative}", flush=True)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(16 * 1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"SHA-256 mismatch: {path}")
            entry["sha256"] = expected
        entry.update(path=str(path.resolve()), bytes=size)
        report[name] = entry
        print(f"OK: {relative}", flush=True)
    return report


def download_configs(root: Path):
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    selected = []
    for entry in api.list_repo_tree(OFFICIAL_REPO, revision=OFFICIAL_REVISION, recursive=True):
        path = entry.path
        if not hasattr(entry, "size"):
            continue
        if path not in ("model_index.json", "LICENSE", "README.md") and not path.startswith(
            ("FL2VA/", "Ref2VA/")
        ):
            continue
        if path.endswith((".safetensors", ".bin", ".pt", ".pth", ".safetensors.index.json")):
            continue
        if entry.size > 20 * 1024 * 1024:
            raise ValueError(f"Refusing unexpected large config file: {path}")
        selected.append(path)
        hf_hub_download(OFFICIAL_REPO, path, revision=OFFICIAL_REVISION, local_dir=root)
    for partition in ("FL2VA", "Ref2VA"):
        for component in ("transformer", "text_encoder", "video_vae", "audio_vae"):
            if not (root / partition / component / "config.json").is_file():
                raise ValueError(f"Missing configuration: {partition}/{component}")
        index = json.loads((root / partition / "model_index.json").read_text())
        if index["_class_name"] != "MiniMaxH3Pipeline":
            raise ValueError("Unsupported official model index")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", type=Path, required=True, help="Directory containing diffusion_models, etc."
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="SHA-256 and header checks; no downloads or writes"
    )
    args = parser.parse_args()
    models = args.models.resolve()
    report = verify(models)
    if args.verify_only:
        print("All five weights match the pinned upstream SHA-256 values.")
        return
    prepared = models / "prepared"
    configs = prepared / "config"
    print("Downloading only official configuration/tokenizer files...", flush=True)
    downloaded = download_configs(configs)
    variants = {}
    for variant in ("fl2va", "ref2va"):
        variants[variant] = {
            component: report[component]["path"] for component in ("text_encoder", "video_vae", "audio_vae")
        }
        variants[variant]["transformer"] = report[variant]["path"]
    manifest = {
        "schema_version": 1,
        "prepared_at": time.time(),
        "config_root": str(configs),
        "weights_revision": WEIGHTS_REVISION,
        "official_revision": OFFICIAL_REVISION,
        "config_files": downloaded,
        "verified_weights": report,
        "variants": variants,
    }
    temporary = prepared / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(prepared / "manifest.json")
    print(f"Prepared: {prepared / 'manifest.json'}")
    print("GPU loading and generation must still be validated on the DGX; this is not a GPU acceptance test.")


if __name__ == "__main__":
    main()
