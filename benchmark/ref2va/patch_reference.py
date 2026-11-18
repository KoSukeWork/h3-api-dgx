"""Hash-guarded opt-in LightX2V match geometry for pinned SGLang only."""

import argparse
import hashlib
from pathlib import Path

ROOT = Path("/opt/h3-api/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3")
BEFORE = {
    "reference_encoding.py": "7b32be64299149074ca12dbab246f6f5fc6bcf12255cbc71ad16f9ea927308fe",
    "prequeue.py": "91652d61cf0a7c552ccf166cbfcb9877f6f2ef2186ff078dd0b4b9fa105db4c1",
}

MATCH_BRANCH = '''    if os.environ.get("H3_REF2VA_REFERENCE_POLICY") == "lightx2v-match-v1":
        if canvas_width is None or canvas_height is None:
            raise ValueError("match policy requires the resolved target canvas")
        area = float(canvas_width) * float(canvas_height)
        if (not math.isfinite(area) or canvas_width <= 0 or canvas_height <= 0):
            raise ValueError("match policy requires a positive finite canvas")
        scale = min(1.0, math.sqrt(area / (source_width * source_height)))
        matched_width = _nearest_multiple(source_width * scale, 32)
        matched_height = _nearest_multiple(source_height * scale, 32)
        return {
            "geometry": "reference_image_resolved",
            "shape_policy_version": "lightx2v_match_v1",
            "base_short_edge": min(canvas_width, canvas_height),
            "effective_short_edge": min(matched_width, matched_height),
            "size_mode": "target_area_cap",
            "multiple": 32, "rounding": "nearest", "allow_upscale": False,
            "width": matched_width, "height": matched_height,
        }

'''


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Pinned source anchor missing or ambiguous")
    return source.replace(old, new, 1)


def transform(name, source):
    if name == "reference_encoding.py":
        source = replace_once(
            source,
            "    height: int | float,\n) -> dict[str, Any]:",
            "    height: int | float,\n    canvas_width: int | None = None,\n"
            "    canvas_height: int | None = None,\n) -> dict[str, Any]:",
        )
        anchor = "    scale = MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE / min(source_width, source_height)"
        return replace_once(source, anchor, MATCH_BRANCH + anchor)
    if name == "prequeue.py":
        anchor = ("                resolved = minimax_h3_resolve_reference_image_shape(\n"
                  "                    width=width,\n                    height=height,\n")
        return replace_once(source, anchor, anchor +
                            "                    canvas_width=int(shape[\"width\"]),\n"
                            "                    canvas_height=int(shape[\"height\"]),\n")
    raise ValueError("Unknown patch target")


def patch(root):
    # Validate both files before writing either. Never accept a marker alone.
    changes = []
    for name, digest in BEFORE.items():
        path = root / name
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"Unsupported source hash: {path}")
        changes.append((path, transform(name, raw.decode()).encode()))
    for path, raw in changes:
        path.write_bytes(raw)
        print(f"Patched {path.name}: {hashlib.sha256(raw).hexdigest()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    patch(parser.parse_args().root)
