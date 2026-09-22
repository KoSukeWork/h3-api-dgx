"""Read-only BF16 QKV layout audit; no torch, CUDA, or whole-shard downloads.

Compare selected rows of a local Comfy checkpoint with the pinned official
checkpoint using bounded HTTP Range requests. Exact byte matches distinguish
head-interleaved QKV from the already-concatenated Q/K/V layout.
"""

import argparse
import json
import struct
import urllib.error
import urllib.request
from pathlib import Path

REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
BASE = f"https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/{REVISION}"
MAX_READ = 16 * 1024 * 1024


def remote_range(url, start, size):
    if not 0 < size <= MAX_READ:
        raise ValueError("Refusing an unbounded reference download")
    end = start + size - 1
    request = urllib.request.Request(
        f"{url}?h3_audit_range={start}-{end}",
        headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                content_range = response.headers.get("Content-Range", "")
                if response.status != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
                    raise RuntimeError("Server did not honor Range; refusing whole-shard download")
                data = response.read(size + 1)
            break
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
    if len(data) != size:
        raise RuntimeError("Reference range length mismatch")
    return data


def read_header(read):
    size = struct.unpack("<Q", read(0, 8))[0]
    if not 2 <= size <= MAX_READ:
        raise ValueError("Invalid safetensors header length")
    return json.loads(read(8, size)), 8 + size


def grouped_row(concat_row, heads=56, head_dim=128):
    projection, within = divmod(concat_row, heads * head_dim)
    head, channel = divmod(within, head_dim)
    return (head * 3 + projection) * head_dim + channel


def audit(path, partition):
    index_url = f"{BASE}/{partition}/transformer/model.safetensors.index.json"
    with urllib.request.urlopen(index_url, timeout=30) as response:
        data = response.read(MAX_READ + 1)
    if len(data) > MAX_READ:
        raise ValueError("Oversized reference index")
    index = json.loads(data)["weight_map"]
    report = {"checkpoint": str(path), "official_revision": REVISION, "samples": []}
    headers = {}
    with path.open("rb") as stream:
        def local_read(start, size):
            stream.seek(start)
            return stream.read(size)

        local_header, local_base = read_header(local_read)
        for key in ("blocks.0.attn.qkv_proj.weight", "token_refiner.blocks.0.attn.qkv_proj.weight"):
            url = f"{BASE}/{partition}/transformer/{index[key]}"
            if url not in headers:
                headers[url] = read_header(lambda start, size, url=url: remote_range(url, start, size))
            official_header, official_base = headers[url]
            local, official = local_header[key], official_header[key]
            if local["dtype"] != "BF16" or official["dtype"] != "BF16":
                raise ValueError("This audit only supports BF16 tensors")
            if local["shape"] != [21504, 5376] or local["shape"] != official["shape"]:
                raise ValueError("Unexpected H3 QKV tensor shape")
            row_bytes = 5376 * 2
            for row in (128, 7168, 7296, 14336):
                original_row = grouped_row(row)
                reference = remote_range(
                    url, official_base + official["data_offsets"][0] + original_row * row_bytes,
                    row_bytes,
                )
                offset = local_base + local["data_offsets"][0]
                result = {
                    "tensor": key, "official_grouped_row": original_row,
                    "local_concat_row": row,
                    "matches_concat": local_read(offset + row * row_bytes, row_bytes) == reference,
                    "matches_grouped": local_read(offset + original_row * row_bytes, row_bytes) == reference,
                }
                report["samples"].append(result)
                print(json.dumps(result), flush=True)
    samples = report["samples"]
    if all(s["matches_concat"] and not s["matches_grouped"] for s in samples):
        report["sampled_layout"] = "already_concatenated_q_k_v"
    elif all(s["matches_grouped"] and not s["matches_concat"] for s in samples):
        report["sampled_layout"] = "official_head_interleaved"
    else:
        report["sampled_layout"] = "inconclusive"
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--partition", choices=("FL2VA", "Ref2VA"), default="FL2VA")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = audit(args.checkpoint, args.partition)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
