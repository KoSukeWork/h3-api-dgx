"""Pinned non-quantized FL2VA Turbo adapter; no user-supplied server file paths."""

import hashlib
import re

LORA_FILENAME = "minimax_h3_turbo_v4_step600_ema.safetensors"
LORA_BYTES = 779849816
LORA_SHA256 = "5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3"


def verify_lora(path):
    if path.stat().st_size != LORA_BYTES:
        raise ValueError("Turbo LoRA size mismatch; install the pinned Larry v4 adapter")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != LORA_SHA256:
        raise ValueError("Turbo LoRA SHA256 mismatch")


def verify_activation(log, path):
    for line in log.splitlines():
        if "LoRA adapter(s)" not in line or str(path.parent) not in line:
            continue
        match = re.search(r"applied to (\d+) layers .*strengths: 1\.0+, merge_mode=dynamic", line)
        if match and int(match[1]) > 0:
            return int(match[1])
    raise RuntimeError("Turbo LoRA was not confirmed active; refusing to generate")
