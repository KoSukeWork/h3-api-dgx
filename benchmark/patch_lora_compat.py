"""Patch the pinned, QKV-fixed H3 model only in the derived Turbo test image."""

import argparse
import hashlib
from pathlib import Path

SOURCE_SHA256 = "8453b66bbe5d89f891ea0f87b8a8f274d514f4d6798e1451046e2d6e8a73ea58"
TARGET = "python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py"
ORIGINAL = """def _accepts_mxfp8_input(linear: nn.Module) -> bool:
    return linear.quant_method is not None and linear.quant_method.accepts_mxfp8_input(
        linear
    )
"""
REPLACEMENT = """def _accepts_mxfp8_input(linear: nn.Module) -> bool:
    # H3 benchmark compatibility: LoRA wrappers need ordinary tensor inputs.
    # Do not unwrap base_layer or proxy its quant_method: that could select a
    # prequantized tuple input that bypasses/breaks the dynamic adapter path.
    quant_method = getattr(linear, "quant_method", None)
    return quant_method is not None and quant_method.accepts_mxfp8_input(linear)
"""


def patched_source(source):
    if source.count(REPLACEMENT) == 1:
        original = source.replace(REPLACEMENT, ORIGINAL, 1)
        if hashlib.sha256(original.encode()).hexdigest() == SOURCE_SHA256:
            return source
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Unexpected H3 source; requires pinned SGLang with our existing BF16 QKV fix")
    if source.count(ORIGINAL) != 1:
        raise ValueError("LoRA compatibility anchor must occur exactly once")
    result = source.replace(ORIGINAL, REPLACEMENT, 1)
    compile(result, TARGET, "exec")  # Syntax only: no torch/CUDA import.
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
    print("Verified H3 LoRA MXFP8 capability-guard fix; existing QKV fix preserved.", flush=True)


if __name__ == "__main__":
    main()
