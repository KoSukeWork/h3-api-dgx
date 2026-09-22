import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark import assets
from benchmark import run as bench
from h3_api.backend import SGLangBackend
from h3_api.settings import Settings


@pytest.mark.parametrize("nfe", [4, 8, 49])
def test_request_uses_sigma_points_without_quality_shortcuts(nfe):
    payload = bench.request_payload(nfe)
    assert payload["num_inference_steps"] == nfe + 1
    assert payload["quality"] == "lossless"
    assert payload["target"] == {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5}
    assert payload["seed"] == 42


@pytest.mark.parametrize("mode", ["turbo", "int8"])
def test_command_keeps_original_and_isolates_candidate(mode):
    original = ["sglang", "serve", "--component-weights-paths.transformer", "/base/bf16"]
    candidate = Path("/candidate/weights.safetensors")
    command = bench.extend_command(original, mode, candidate)
    assert original[-1] == "/base/bf16"
    assert "--quantization" not in command
    assert command[command.index("--warmup-mode") + 1] == "off"
    if mode == "turbo":
        assert command[3] == "/base/bf16"
        assert command[command.index("--lora-merge-mode") + 1] == "dynamic"
        assert command[command.index("--lora-weight-name") + 1] == candidate.name
    else:
        assert command[3] == str(candidate)
        assert "--lora-path" not in command


@pytest.mark.parametrize("mode", ["turbo", "int8"])
def test_env_does_not_apply_bf16_override_to_int8(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(bench, "ROOT", tmp_path / "bench")
    monkeypatch.setattr(
        SGLangBackend,
        "launch_context",
        lambda *args: {
            "cwd": "example",
            "env": {"H3_COMFY_BF16_QKV_LAYOUT": "concatenated"},
        },
    )
    backend = bench.BenchBackend(Settings(tmp_path, tmp_path, "key"), mode, tmp_path / "weights")
    try:
        env = backend.launch_context("fl2va")["env"]
        assert ("H3_COMFY_BF16_QKV_LAYOUT" in env) == (mode == "turbo")
        assert env["SGLANG_DIFFUSION_HOST_SPILL_DIR"] == str(tmp_path / "bench" / "cache" / mode / "spill-v1")
    finally:
        asyncio.run(backend.close())


def test_actual_steps_and_pipeline_timing_required():
    log = (
        "infer_steps: 9\n\x1b[32mminimax_h3 denoise: 100%|xxx| 8/8 [06:00]\x1b[0m\n"
        "[MiniMaxH3DenoisingStage] finished in 360.0000 seconds\n"
        "Pixel data generated successfully in 410.00 seconds"
    )
    assert bench.stage_stats(log, 8)["pipeline_seconds"] == 410
    with pytest.raises(RuntimeError, match="sigma-point"):
        bench.stage_stats(log, 4)
    with pytest.raises(RuntimeError, match="complete"):
        bench.stage_stats(log.replace("8/8", "7/8"), 8)
    with pytest.raises(RuntimeError, match="inactive"):
        bench.stage_stats(log + "\nLoRA adapter is set, but not effective", 8)


def test_lora_requires_positive_application_not_just_loading():
    asset = Path("/assets/turbo/weights.safetensors")
    log = (
        f"Rank 0: LoRA adapter(s) {asset.parent} applied to 520 layers "
        "(targets: all, strengths: 1.00, merge_mode=dynamic)"
    )
    assert bench.lora_activation(log, asset)["applied_layers"] == 520
    for invalid in ("loaded LoRA adapter", log.replace("520", "0"), log.replace("1.00", "0.00")):
        with pytest.raises(RuntimeError, match="application"):
            bench.lora_activation(invalid, asset)


def test_asset_hash_checks_bytes_not_just_length(tmp_path, monkeypatch):
    folder = tmp_path / "test"
    folder.mkdir()
    path = folder / "weight"
    path.write_bytes(b"abc")
    monkeypatch.setitem(
        assets.ASSETS,
        "test",
        {
            "file": "weight",
            "bytes": 3,
            "sha256": hashlib.sha256(b"abc").hexdigest(),
        },
    )
    assert assets.verify("test", tmp_path) == path
    path.write_bytes(b"xyz")
    with pytest.raises(ValueError, match="SHA256"):
        assets.verify("test", tmp_path)


def test_x86_guard_never_calls_gpu(monkeypatch):
    monkeypatch.setattr(bench.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(bench.subprocess, "check_output", lambda *a, **k: pytest.fail("GPU call on x86"))
    with pytest.raises(RuntimeError, match="x86"):
        bench.require_dgx_idle()


def test_timeout_cancels_request():
    async def check():
        cancelled = asyncio.Event()

        async def generate(*args):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        backend = SimpleNamespace(settings=SimpleNamespace(task_timeout=0.01), generate=generate)
        with pytest.raises(TimeoutError):
            await bench.timed_generate(backend, {}, Path("example.mp4"))
        assert cancelled.is_set()

    asyncio.run(check())
