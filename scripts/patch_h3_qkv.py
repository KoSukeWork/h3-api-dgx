"""Apply the narrowly scoped Comfy BF16 QKV compatibility patch at image build.

The gateway enables it only for the two hash-verified Comfy BF16 checkpoints.
Native checkpoints and quantized loaders retain the upstream behavior.
"""

import argparse
import hashlib
from pathlib import Path

SOURCE_SHA256 = "2a61d5c8b0418eed72fd3443cc47b6a06ed8512f6c834b608aa4e3e55e7489fa"
TARGET = "python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py"
ORIGINAL = """        checkpoint_qkv_is_native = (
            checkpoint_qkv_is_native or arch.checkpoint_uses_diffusers_layout
        )
"""
REPLACEMENT = """        # H3 API compatibility: verified Comfy BF16 exports already concatenate
        # Q, K, V. Do not apply the official head-interleaved row permutation twice.
        # This override does not change quantized or Diffusers checkpoint handling.
        checkpoint_qkv_is_native = (
            checkpoint_qkv_is_native
            or arch.checkpoint_uses_diffusers_layout
            or (
                quant_config is None
                and os.environ.get("H3_COMFY_BF16_QKV_LAYOUT") == "concatenated"
            )
        )
"""


def patched_source(source):
    if source.count(REPLACEMENT) == 1:
        original = source.replace(REPLACEMENT, ORIGINAL, 1)
        if hashlib.sha256(original.encode()).hexdigest() == SOURCE_SHA256:
            return source
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Unexpected SGLang H3 source; refusing to patch another revision")
    if source.count(ORIGINAL) != 1:
        raise ValueError("QKV patch anchor must occur exactly once")
    result = source.replace(ORIGINAL, REPLACEMENT, 1)
    compile(result, TARGET, "exec")  # Syntax only; no torch/CUDA import or inference.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang", type=Path, default=Path("/opt/h3-api/sglang"))
    args = parser.parse_args()
    path = args.sglang / TARGET
    original = path.read_text(encoding="utf-8")
    updated = patched_source(original)
    if updated != original:
        path.write_text(updated, encoding="utf-8", newline="\n")
    print("Verified H3 Comfy BF16 QKV compatibility patch installed.", flush=True)


if __name__ == "__main__":
    main()
