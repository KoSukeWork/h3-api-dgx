"""Isolated, serial H3 API benchmark. Run only on DGX via benchmark/compose.yaml."""

import argparse
import asyncio
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, "/opt/h3-api/app")

from filelock import FileLock

from h3_api.backend import SGLangBackend
from h3_api.settings import SGLANG_REVISION, Settings

try:
    from .assets import ASSETS, verify
except ImportError:
    from assets import ASSETS, verify

ROOT = Path("/srv/h3/data/benchmark")
PROMPT = "A red toy car drives across a white table. Camera: static. Audio: quiet electric motor."
INFERENCE_PYTHON = "/opt/h3-api/inference-venv/bin/python"


def request_payload(nfe, prompt=PROMPT, seed=42):
    return {
        "task": "t2va",
        "prompt": prompt,
        "conditions": [],
        "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5},
        "seed": seed,
        "quality": "lossless",
        "num_inference_steps": nfe + 1,
    }


def extend_command(command, mode, asset):
    if mode not in ASSETS:
        raise ValueError(f"Unknown benchmark mode: {mode}")
    command = list(command)
    # Replace the default three startup probes with an explicit, recorded warmup.
    command += ["--warmup-mode", "off"]
    if mode == "turbo":
        command += [
            "--lora-path",
            str(asset.parent),
            "--lora-weight-name",
            asset.name,
            "--lora-nickname",
            "h3-bench-larry-v4",
            "--lora-scale",
            "1.0",
            # Runtime adapter application avoids writing merged updates into mmap weights.
            "--lora-merge-mode",
            "dynamic",
        ]
    else:
        index = command.index("--component-weights-paths.transformer") + 1
        command[index] = str(asset)
        # Serialized ConvRot is self-describing: never add an online quantization flag.
    return command


class BenchBackend(SGLangBackend):
    def __init__(self, settings, mode, asset):
        super().__init__(settings)
        self.mode, self.asset = mode, asset

    def command(self, variant):
        return extend_command(super().command(variant), self.mode, self.asset)

    def launch_context(self, variant=None):
        context = super().launch_context(variant)
        cache = ROOT / "cache" / self.mode
        for key, directory in (
            ("HF_HOME", cache / "hf"),
            ("SGLANG_DIFFUSION_CACHE_ROOT", cache / "sglang"),
            ("SGLANG_DIFFUSION_HOST_SPILL_DIR", cache / "spill-v1"),
            ("TRITON_CACHE_DIR", cache / "triton"),
            ("TORCH_HOME", cache / "torch"),
            ("CUDA_CACHE_PATH", cache / "cuda"),
        ):
            directory.mkdir(parents=True, exist_ok=True)
            context["env"][key] = str(directory)
        if self.mode == "int8":
            # Serialized KitchenInt8Config already declares concatenated QKV.
            # The BF16-only override must not be applied to this different file.
            context["env"].pop("H3_COMFY_BF16_QKV_LAYOUT", None)
        return context


def lora_activation(log, asset):
    lines = [line for line in log.splitlines() if "LoRA adapter(s)" in line and str(asset.parent) in line]
    for line in lines:
        match = re.search(r"applied to (\d+) layers .*strengths: 1\.0+, merge_mode=dynamic", line)
        if match and int(match[1]) > 0:
            return {"applied_layers": int(match[1]), "evidence": line}
    raise RuntimeError("No positive LoRA application evidence; do not compare this run as Turbo")


def stage_stats(log, expected_nfe):
    clean = re.sub(r"\x1b\[[0-9;]*m", "", log)
    if "LoRA adapter is set, but not effective" in clean:
        raise RuntimeError("LoRA was inactive during generation")
    steps = re.findall(r"infer_steps:\s*(\d+)", clean)
    progress = re.findall(r"minimax_h3 denoise:.*?(\d+)/(\d+)", clean)
    if not steps or int(steps[-1]) != expected_nfe + 1:
        raise RuntimeError("Backend did not log the requested sigma-point count")
    if not any(int(a) == int(b) == expected_nfe for a, b in progress):
        raise RuntimeError("Backend did not complete the requested NFE count")
    totals = re.findall(r"Pixel data generated successfully in ([\d.]+) seconds", clean)
    if not totals:
        raise RuntimeError("Missing pipeline completion timing")
    stages = dict(re.findall(r"\[(MiniMaxH3\w+Stage)\] finished in ([\d.]+) seconds", clean))
    return {"pipeline_seconds": float(totals[-1]), "stages_seconds": {k: float(v) for k, v in stages.items()}}


def media_check(path):
    raw = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    info = json.loads(raw)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    audio = next(s for s in info["streams"] if s["codec_type"] == "audio")
    if (video["width"], video["height"]) != (1344, 768):
        raise RuntimeError("Unexpected resolution: benchmark must stay native 1344x768")
    if int(video.get("nb_frames", 0)) != 124:
        raise RuntimeError("Unexpected frame count: this comparison requires 124 frames")
    from fractions import Fraction

    if Fraction(video["r_frame_rate"]) != 24 or audio["channels"] != 2 or int(audio["sample_rate"]) != 32000:
        raise RuntimeError("Unexpected frame rate or audio format")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        check=True,
    )
    frames = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            "fps=1,scale=32:32",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ]
    )
    means = [sum(frames[i : i + 1024]) / len(frames[i : i + 1024]) for i in range(0, len(frames), 1024)]
    if not means or max(means) <= 1:
        raise RuntimeError("Sampled frames are black; this is not a valid quality result")
    info["sample_frame_means"] = means
    return info


def require_dgx_idle():
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("No x86 inference: run on the DGX using Compose")
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    if "GB10" not in gpu:
        raise RuntimeError(f"Expected GB10, found {gpu}")
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    if processes:
        raise RuntimeError(f"GPU compute process already present; stop the API first:\n{processes}")


async def timed_generate(backend, payload, path):
    started = time.monotonic()
    task = asyncio.create_task(backend.generate(payload, path))
    try:
        async with asyncio.timeout(backend.settings.task_timeout):
            while not task.done():
                await asyncio.wait({task}, timeout=20)
                print(f"{path.stem}: {time.monotonic() - started:.0f}s elapsed", flush=True)
            await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    return time.monotonic() - started


async def run(args):
    require_dgx_idle()
    # Exact hashes, including INT8, are verified before model loading and timing.
    asset = verify(args.mode)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{args.mode}-{os.getpid()}"
    directory = ROOT / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    settings = Settings(directory, Path("/srv/h3/models"), "benchmark-local-only")
    backend = BenchBackend(settings, args.mode, asset)
    log_path = directory / "logs" / "sglang-fl2va.log"
    summary = {
        "run_id": run_id,
        "mode": args.mode,
        "asset": ASSETS[args.mode],
        "request_geometry": request_payload(8, args.prompt, args.seed),
        "results": [],
        "status": "running",
        "warmup_policy": "4-NFE warmup timed separately; startup probes off",
        "baseline_reference_seconds": 2103.52,
        "sglang_revision": SGLANG_REVISION,
        "gpu_and_driver": (
            await asyncio.to_thread(
                subprocess.check_output,
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                text=True,
            )
        ).strip(),
        "baseline_note": "Historical single BF16 run, not a repeated controlled benchmark; residency may differ.",
        "baseline_comparable": args.prompt == PROMPT and args.seed == 42,
        "quality_note": "Media checks do not establish visual or audio quality. Inspect the MP4s.",
    }

    def save():
        (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    save()
    resolved = Path("/opt/h3-api/inference-resolved.txt")
    if resolved.is_file():
        (directory / "base-inference-resolved.txt").write_bytes(resolved.read_bytes())
    print(f"RESULT DIRECTORY: {directory}", flush=True)
    try:
        if args.mode == "int8":
            with (directory / "int8-kernel-gate.log").open("w") as gate_log:
                proc = await asyncio.create_subprocess_exec(
                    INFERENCE_PYTHON,
                    "/bench/probe_int8.py",
                    stdout=gate_log,
                    stderr=asyncio.subprocess.STDOUT,
                )
                try:
                    code = await asyncio.wait_for(proc.wait(), 180)
                finally:
                    if proc.returncode is None:
                        proc.kill()
                        await proc.wait()
                if code != 0:
                    raise RuntimeError("INT8 kernel gate failed; see int8-kernel-gate.log")
        summary["command"] = backend.command("fl2va")
        started = time.monotonic()
        await backend.ensure("fl2va")
        summary["load_to_ready_seconds"] = time.monotonic() - started
        if args.mode == "turbo":
            summary["lora_activation"] = lora_activation(log_path.read_text(errors="replace"), asset)
        else:
            backend.log.write(
                b"\nBENCHMARK: transformer is SHA-verified serialized INT8 ConvRot; "
                b"the inherited BF16 banner describes the base manifest, NOT this transformer.\n"
            )
        save()
        print(
            f"Backend ready in {summary['load_to_ready_seconds']:.1f}s. Explicit warmup begins.", flush=True
        )
        profiles = [("warmup-4", 4)] + (
            [("turbo-8", 8), ("turbo-4", 4)] if args.mode == "turbo" else [("int8-49", 49)]
        )
        for label, nfe in profiles:
            payload = request_payload(nfe, args.prompt, args.seed)
            (directory / f"{label}.request.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            offset = log_path.stat().st_size
            path = directory / f"{label}.mp4"
            elapsed = await timed_generate(backend, payload, path)
            with log_path.open("rb") as stream:
                stream.seek(offset)
                excerpt = stream.read().decode("utf-8", errors="replace")
            stats = stage_stats(excerpt, nfe)
            info = media_check(path)
            record = {
                "label": label,
                "nfe": nfe,
                "measured": not label.startswith("warmup"),
                "api_wall_seconds": elapsed,
                "file": str(path),
                **stats,
            }
            (directory / f"{label}.media.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            if record["measured"] and summary["baseline_comparable"]:
                record["speedup_vs_previous_baseline"] = 2103.52 / stats["pipeline_seconds"]
            summary["results"].append(record)
            save()
            print(json.dumps(record, indent=2), flush=True)
        summary["status"] = "completed"
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            await backend.close()
        finally:
            save()
            print(f"SUMMARY: {directory / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("turbo", "int8"), default="turbo")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        parser.error("Run on the ARM64 DGX only; no local x86 inference")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    task = loop.create_task(run(args))
    if sys.platform == "linux":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, task.cancel)
    try:
        ROOT.mkdir(parents=True, exist_ok=True)
        # Shared bind mount lock prevents two benchmark containers racing at startup.
        with FileLock(str(ROOT / "gpu-benchmark.lock"), timeout=0):
            loop.run_until_complete(task)
    finally:
        if not task.done():
            task.cancel()
            loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
        loop.close()


if __name__ == "__main__":
    main()
