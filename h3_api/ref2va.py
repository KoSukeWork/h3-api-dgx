"""Pinned, GPU-validated Ref2VA adapter and native configuration routing."""

import hashlib
from pathlib import Path

from h3_api.turbo import verify_activation

FILENAME = "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors"
BYTES = 1383677768
SHA256 = "9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99"
RECIPE_KEY = "lightx2v-ref2va-v01-" + SHA256[:12] + "-alpha8-scale1-match-v1-qkvconcat-v1"


def verify_lora(path):
    if path.stat().st_size != BYTES:
        raise ValueError("Ref2VA LoRA size mismatch")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != SHA256:
        raise ValueError("Ref2VA LoRA SHA256 mismatch")


def verify_active(log, path):
    if verify_activation(log, path) != 208:
        raise RuntimeError("Expected exactly 208 active LightX2V Ref2VA modules")


def route(command, view):
    command = list(command)
    source = Path(command[command.index("--model-path") + 1]).resolve()
    view.mkdir(parents=True, exist_ok=True)
    for name in ("model_index.json", "transformer", "text_encoder", "video_vae",
                 "audio_vae", "processor", "tokenizer", "Ref2VA"):
        target = source if name == "Ref2VA" else source / name
        if not target.exists():
            raise FileNotFoundError(target)
        link = view / name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise ValueError("Unexpected existing Ref2VA configuration link")
        elif link.exists():
            raise ValueError("Unexpected existing Ref2VA configuration view")
        else:
            link.symlink_to(target, target_is_directory=target.is_dir())
    command[command.index("--model-path") + 1] = str(view)
    command[command.index("--model-subfolder") + 1] = "Ref2VA"
    return command + ["--model-variant", "ref2va"]
