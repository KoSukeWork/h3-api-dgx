"""One cold Ref2VA image test, ARM64 GB10 only; no production mutations."""

import argparse
import asyncio
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from filelock import FileLock

from benchmark.ref2va.assets import SPEC, verify
from benchmark.run import require_dgx_idle, stage_stats, timed_generate
from h3_api.backend import SGLangBackend
from h3_api.settings import Settings

ROOT = Path("/srv/h3/ref2va-test")


def extend_command(command, variant, asset):
    if variant != "ref2va":
        raise ValueError("LightX2V Ref2VA adapter cannot be applied to FL2VA")
    return list(command) + [
        "--warmup-mode", "off", "--lora-path", str(asset.parent),
        "--lora-weight-name", asset.name, "--lora-nickname", "lightx2v-ref2va-v01",
        "--lora-alpha", "8", "--lora-scale", "1.0", "--lora-merge-mode", "dynamic",
    ]


def route_ref2va(command, view):
    command = list(command)
    command[command.index("--model-path") + 1] = str(view)
    command[command.index("--model-subfolder") + 1] = "Ref2VA"
    return command + ["--model-variant", "ref2va"]


def config_view(source, view):
    # Registry discovery needs a native index at the root, while semantic
    # model-variant routing appends Ref2VA. Link both to the same immutable
    # config partition without altering the prepared shared model directory.
    view.mkdir(parents=True, exist_ok=True)
    for name in ("model_index.json", "transformer", "text_encoder", "video_vae",
                 "audio_vae", "processor", "tokenizer", "Ref2VA"):
        target = source if name == "Ref2VA" else source / name
        link = view / name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise ValueError("Unexpected existing configuration link")
        elif link.exists():
            raise ValueError("Unexpected existing configuration view")
        else:
            link.symlink_to(target, target_is_directory=target.is_dir())


def activation(log, asset):
    for line in log.splitlines():
        if "LoRA adapter(s)" not in line or str(asset.parent) not in line:
            continue
        match = re.search(r"applied to (\d+) layers .*strengths: 1\.0+, merge_mode=dynamic", line)
        if match and int(match[1]) == 208:
            return {"applied_layers": 208, "evidence": line}
    raise RuntimeError("Expected exactly 208 active Ref2VA LoRA layers at strength 1.0/dynamic")


class RefBackend(SGLangBackend):
    def __init__(self, settings, asset):
        if settings.turbo_lora is not None:
            raise ValueError("Do not combine Larry and Ref2VA adapters")
        super().__init__(settings)
        self.asset = asset

    def command(self, variant):
        command = super().command(variant)
        source = Path(command[command.index("--model-path") + 1])
        view = self.settings.data_dir / "config-view"
        config_view(source, view)
        return extend_command(route_ref2va(command, view), variant, self.asset)

    def launch_context(self, variant=None):
        context = super().launch_context(variant)
        context["env"]["H3_REF2VA_REFERENCE_POLICY"] = "lightx2v-match-v1"
        cache = ROOT / "cache" / "ref2va-v01-match-alpha8-qkvconcat-v1"
        for key, leaf in (("HF_HOME", "hf"), ("SGLANG_DIFFUSION_CACHE_ROOT", "sglang"),
                          ("SGLANG_DIFFUSION_HOST_SPILL_DIR", "spill"),
                          ("TRITON_CACHE_DIR", "triton"), ("TORCH_HOME", "torch"),
                          ("CUDA_CACHE_PATH", "cuda")):
            path = cache / leaf
            path.mkdir(parents=True, exist_ok=True)
            context["env"][key] = str(path)
        return context


def request_payload(case, image, prompt_variant):
    # First smoke test deliberately differs from the 544p training recipe.
    return {
        "task": "ref2va", "prompt": case[f"prompt_{prompt_variant}"],
        "conditions": [{"type": "image", "role": "reference", "uri": str(image)}],
        "target": {"short_edge": 480, "aspect_ratio": "16:9", "duration_seconds": 5},
        "seed": int(case["seed"]), "quality": "lossless", "num_inference_steps": 5,
        "flow_shift": 12.0, "audio_flow_shift": 3.0,
    }


def media_check(path):
    from fractions import Fraction

    info = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        text=True,
    ))
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    audio = next(s for s in info["streams"] if s["codec_type"] == "audio")
    if (video["width"], video["height"], int(video.get("nb_frames", 0))) != (864, 480, 124):
        raise RuntimeError("Unexpected smoke-test video geometry/frame count")
    if Fraction(video["r_frame_rate"]) != 24 or audio["channels"] != 2 or int(audio["sample_rate"]) != 32000:
        raise RuntimeError("Unexpected smoke-test video/audio rates")
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path),
                    "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"], check=True)
    return info


async def run(args):
    require_dgx_idle()
    print("Verifying Ref2VA adapter SHA256 before loading...", flush=True)
    asset = verify()
    case = json.loads(args.case.read_text(encoding="utf-8"))
    image = (args.case.parent / case["image"]).resolve()
    if image.parent != args.case.parent.resolve():
        raise ValueError("Reference must be directly inside the private case directory")
    if hashlib.sha256(image.read_bytes()).hexdigest() != case["image_sha256"]:
        raise ValueError("Private reference image hash mismatch")
    payload = request_payload(case, image, args.prompt_variant)
    name = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{args.prompt_variant}-{os.getpid()}"
    directory = ROOT / "runs" / name
    directory.mkdir(parents=True, exist_ok=False)
    backend = RefBackend(Settings(directory, Path("/srv/h3/models"), "isolated-test-only"), asset)
    log_path = directory / "logs" / "sglang-ref2va.log"
    summary = {"status": "running", "adapter": SPEC, "prompt_variant": args.prompt_variant,
               "warmup": "none; cold first generation, not steady-state benchmark",
               "reference_policy": "lightx2v-match-v1", "alpha": 8, "rank": 128, "scale": 1.0,
               "image_sha256": case["image_sha256"], "nfe": 4,
               "quality_note": "Media validation does not establish visual or audio quality"}

    def save():
        (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"RESULT DIRECTORY: {directory}", flush=True)
    (directory / "request.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    save()
    try:
        summary["command"] = backend.command("ref2va")
        started = time.monotonic()
        await backend.ensure("ref2va")
        summary["load_to_ready_seconds"] = time.monotonic() - started
        summary["activation"] = activation(log_path.read_text(errors="replace"), asset)
        save()
        print(f"Ref2VA ready in {summary['load_to_ready_seconds']:.1f}s; generating one cold sample", flush=True)
        offset = log_path.stat().st_size
        output = directory / "ref2va-4.mp4"
        summary["api_wall_seconds"] = await timed_generate(backend, payload, output)
        with log_path.open("rb") as stream:
            stream.seek(offset)
            summary.update(stage_stats(stream.read().decode(errors="replace"), 4))
        info = media_check(output)
        (directory / "media.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        summary.update(status="completed", file=str(output))
    except BaseException as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            await backend.close()
        finally:
            save()
            print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=Path("/case/case.json"))
    parser.add_argument("--prompt-variant", choices=("original", "reference_bound"), default="original")
    args = parser.parse_args()
    require_dgx_idle()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    task = loop.create_task(run(args))
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, task.cancel)
    try:
        with FileLock(str(ROOT / "gpu-test.lock"), timeout=0):
            loop.run_until_complete(task)
    finally:
        if not task.done():
            task.cancel()
            loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
        loop.close()


if __name__ == "__main__":
    main()
