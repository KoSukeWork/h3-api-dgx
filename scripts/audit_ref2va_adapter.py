"""CPU-only, bounded header audit. Never downloads full weights or imports torch.

Run from the repository root: python -m scripts.audit_ref2va_adapter
Upstream source and LFS hashes are evidence, not GPU compatibility certification.
"""

import argparse
import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts.audit_h3_qkv_layout import read_header, remote_range

REPO = "lightx2v/Minimax-h3-Turbo"
REVISION = "30f485ed874aa44c8183348ad361c5eabad318da"
FILES = {
    "diffusers-v0.1": {
        "file": "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors",
        "bytes": 1383677768,
        "sha256": "9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99",
    },
    "comfy-v0.1": {
        "file": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        "bytes": 1956193000,
        "sha256": "5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c",
    },
}


def summarize(header):
    tensors = {name: spec for name, spec in header.items() if name != "__metadata__"}
    ranks = Counter()
    targets = Counter()
    for name, spec in tensors.items():
        if re.search(r"\.lora_(A|down)(\.default)?\.weight$", name):
            ranks[str(spec["shape"][0])] += 1
        normalized = re.sub(r"\.(\d+)(?=\.)", ".N", name)
        targets[normalized] += 1
    return {
        "metadata": header.get("__metadata__", {}),
        "tensor_count": len(tensors),
        "dtype_counts": dict(Counter(spec["dtype"] for spec in tensors.values())),
        "a_rank_counts": dict(ranks),
        "target_patterns": dict(sorted(targets.items())),
        "sample_shapes": {name: tensors[name]["shape"] for name in sorted(tensors)[:16]},
    }


def audit_mapping(header, config_source):
    """Check PEFT names/shapes against the inspected upstream literal mapping.

    Static analysis only: never import or execute downloaded source. This cannot
    certify floating-point transforms, CUDA kernels, or actual adapter activation.
    """
    tree = ast.parse(config_source)
    mapping = None
    for item in ast.walk(tree):
        if (isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
                and item.target.id == "param_names_mapping"):
            mapping = ast.literal_eval(item.value.keywords[0].value.body)
            break
    if mapping is None:
        raise ValueError("Upstream param_names_mapping literal missing")
    groups = defaultdict(dict)
    for source_name, spec in header.items():
        if source_name == "__metadata__":
            continue
        if spec["dtype"] != "BF16":
            raise ValueError("Expected the original BF16 PEFT adapter, not a converted adapter")
        name = re.sub(r"(\.lora_[AB])\.default\.weight$", r"\1", source_name)
        applied = set()
        section = None
        for _ in range(max(8, len(mapping) * 2)):
            for pattern, replacement in mapping.items():
                if pattern in applied or not re.match(pattern, name):
                    continue
                target = replacement[0] if isinstance(replacement, tuple) else replacement
                updated = re.sub(pattern, target, name)
                if updated == name:
                    continue
                if isinstance(replacement, tuple):
                    section = replacement[1]
                    if replacement[2] != 3:
                        raise ValueError("Unexpected fused section count")
                applied.add(pattern)
                name = updated
                break
            else:
                break
        match = re.fullmatch(
            r"((?:token_refiner\.)?blocks\.\d+\.(?:attn\.(?:qkv_proj|out_proj)|mlp\.fc[12]))\.lora_([AB])",
            name,
        )
        if not match:
            raise ValueError(f"Unmapped adapter tensor: {source_name}")
        module, side = match.groups()
        slot = (side, section)
        if slot in groups[module]:
            raise ValueError(f"Duplicate mapped tensor: {source_name}")
        groups[module][slot] = spec["shape"]
    # Dimensions from the pinned full H3 architecture, not from a pruned model.
    expected_dims = {"qkv_proj": (5376, 7168), "out_proj": (7168, 5376),
                     "fc1": (5376, 28672), "fc2": (14336, 5376)}
    expected_modules = {
        f"{prefix}blocks.{index}.{suffix}"
        for prefix, count in (("", 50), ("token_refiner.", 2))
        for index in range(count)
        for suffix in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
    }
    if set(groups) != expected_modules:
        raise ValueError("Missing or unexpected adapter modules")
    for module, tensors in groups.items():
        kind = module.rsplit(".", 1)[1]
        sections = (0, 1, 2) if kind == "qkv_proj" else (None,)
        if set(tensors) != {(side, part) for side in ("A", "B") for part in sections}:
            raise ValueError(f"Incomplete A/B or fused QKV group: {module}")
        in_dim, out_dim = expected_dims[kind]
        for part in sections:
            if tensors[("A", part)] != [128, in_dim] or tensors[("B", part)] != [out_dim, 128]:
                raise ValueError(f"Unexpected adapter shape: {module}")
    return {"mapped_tensor_count": sum(len(t) for t in groups.values()),
            "native_module_count": len(groups), "complete_qkv_groups": 52,
            "rank": 128, "static_name_shape_check": "passed", "gpu_validated": False}


def audit(name):
    spec = FILES[name]
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{spec['file']}"
    header, data_start = read_header(lambda start, size: remote_range(url, start, size))
    tensors = [value for key, value in header.items() if key != "__metadata__"]
    computed_size = data_start + max(value["data_offsets"][1] for value in tensors)
    if computed_size != spec["bytes"]:
        raise ValueError("Header tensor offsets do not match the pinned file size")
    return {
        "repo": REPO, "revision": REVISION, **spec,
        "full_file_sha256_verified": False,
        "header_bytes_downloaded": data_start,
        "summary": summarize(header),
        "header": header,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist/ref2va-audit/headers.json"))
    args = parser.parse_args()
    report = {name: audit(name) for name in FILES}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, result in report.items():
        print(json.dumps({"name": name, "header_bytes_downloaded": result["header_bytes_downloaded"],
                          **result["summary"]}, indent=2))
    print(f"Report: {args.output}; report SHA256: {hashlib.sha256(args.output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
