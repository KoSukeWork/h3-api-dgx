import ast
import math
import os
from pathlib import Path, PurePosixPath

import pytest

from benchmark.ref2va.patch_reference import MATCH_BRANCH, patch, transform
from benchmark.ref2va.run import activation, extend_command, request_payload, route_ref2va


def geometry(width, height, canvas_width=None, canvas_height=None):
    # Execute the exact injected CPU-only branch without importing torch/SGLang.
    source = ("def resolve(source_width, source_height, canvas_width, canvas_height):\n"
              + MATCH_BRANCH + "    return 'upstream-default'\n")
    namespace = {"math": math, "os": os,
                 "_nearest_multiple": lambda v, n: max(n, round(v / n) * n)}
    exec(compile(ast.parse(source), "<geometry-patch>", "exec"), namespace)  # noqa: S102 - owned pure patch text
    return namespace["resolve"](width, height, canvas_width, canvas_height)


def test_match_policy_is_opt_in_and_requires_canvas(monkeypatch):
    monkeypatch.delenv("H3_REF2VA_REFERENCE_POLICY", raising=False)
    assert geometry(1488, 1017) == "upstream-default"
    monkeypatch.setenv("H3_REF2VA_REFERENCE_POLICY", "lightx2v-match-v1")
    with pytest.raises(ValueError, match="resolved target"):
        geometry(1488, 1017)
    result = geometry(1488, 1017, 864, 480)
    assert (result["width"], result["height"]) == (768, 544)
    assert geometry(320, 320, 864, 480)["width"] == 320
    with pytest.raises(ValueError, match="positive finite"):
        geometry(1488, 1017, 0, 480)


def test_patch_rejects_unknown_source_before_any_write(tmp_path):
    path = tmp_path / "reference_encoding.py"
    path.write_text("unknown")
    with pytest.raises(ValueError, match="Unsupported source hash"):
        patch(tmp_path)
    assert path.read_text() == "unknown"
    with pytest.raises(ValueError, match="anchor"):
        transform("prequeue.py", "unknown")


def test_profile_recipe_and_exact_layer_gate():
    asset = PurePosixPath("/assets/ref2va/adapter.safetensors")
    command = extend_command(["sglang"], "ref2va", asset)
    assert command[command.index("--lora-alpha") + 1] == "8"
    assert command[command.index("--lora-scale") + 1] == "1.0"
    with pytest.raises(ValueError, match="cannot be applied"):
        extend_command([], "fl2va", asset)
    line = "LoRA adapter(s) /assets/ref2va applied to 208 layers (targets: transformer, strengths: 1.0, merge_mode=dynamic)"
    assert activation(line, asset)["applied_layers"] == 208
    with pytest.raises(RuntimeError):
        activation(line.replace("208", "200"), asset)
    case = {"prompt_original": "original", "prompt_reference_bound": "<Picture 1> test", "seed": 42}
    payload = request_payload(case, Path("/case/reference.png"), "original")
    assert payload["prompt"] == "original"
    assert (payload["num_inference_steps"], payload["flow_shift"], payload["audio_flow_shift"]) == (5, 12, 3)
    assert payload["conditions"][0]["role"] == "reference"


def test_semantic_variant_and_registry_paths_are_consistent():
    command = route_ref2va(["sglang", "--model-path", "/prepared/config/Ref2VA",
                           "--model-subfolder", "."], PurePosixPath("/run/config-view"))
    assert command[command.index("--model-path") + 1] == "/run/config-view"
    assert command[command.index("--model-subfolder") + 1] == "Ref2VA"
    assert command[command.index("--model-variant") + 1] == "ref2va"
