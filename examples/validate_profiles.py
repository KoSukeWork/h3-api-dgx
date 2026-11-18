"""DGX acceptance via the production HTTP API: Larry -> Ref2VA -> Larry.

Reads a private case JSON and H3_API_KEY from the container environment.
Writes private request results under the supplied output directory. Never commit them.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import httpx
from client import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        parser.error("Production acceptance must run inside the ARM64 DGX container")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    case = json.loads(args.case.read_text(encoding="utf-8"))
    image = args.case.parent / case["image"]
    if hashlib.sha256(image.read_bytes()).hexdigest() != case["image_sha256"]:
        raise ValueError("Private image SHA mismatch")
    summary = {"status": "running", "results": []}
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=120, trust_env=False,
                      headers={"Authorization": "Bearer " + os.environ["H3_API_KEY"]}) as client:
        try:
            ready = client.get("/ready")
            ready.raise_for_status()
            ready = ready.json()
            if ready["active_job_id"] or any(ready["jobs"].get(s, 0) for s in ("queued", "loading", "running")):
                raise RuntimeError("Production queue is not empty; defer acceptance")
            response = client.get("/v1/profiles")
            response.raise_for_status()
            summary["catalog"] = response.json()
            with image.open("rb") as stream:
                response = client.post("/v1/files", files={"file": (image.name, stream)})
            response.raise_for_status()
            file_id = response.json()["id"]
            reference = {"task": "ref2va", "profile": "lightx2v-ref2va-v0.1",
                         "prompt": case["prompt_original"], "seed": case["seed"],
                         "conditions": [{"type": "image", "role": "reference", "file_id": file_id}],
                         "target": {"short_edge": 480, "aspect_ratio": "16:9", "duration_seconds": 5},
                         "steps": 4}
            # Invalid combinations must fail before queueing or generation.
            rejected = client.post("/v1/videos", json={**reference, "steps": 8})
            if rejected.status_code != 422:
                raise RuntimeError("Ref2VA invalid-step guard failed")
            baseline = {"task": "t2va", "prompt": "A red toy car drives across a white table. Camera: static.",
                        "preset": "fast", "seed": 42,
                        "target": {"short_edge": 480, "aspect_ratio": "16:9", "duration_seconds": 5}}
            for label, payload, expected in (
                ("larry-before", baseline, "larry-turbo-v4"),
                ("lightx-reference", reference, "lightx2v-ref2va-v0.1"),
                ("larry-after", {**baseline, "profile": "larry-turbo-v4"}, "larry-turbo-v4"),
            ):
                started = time.monotonic()
                output = args.output_dir / f"{label}.mp4"
                job_id = generate(client, payload, output)
                elapsed = time.monotonic() - started
                job = client.get(f"/v1/videos/{job_id}").json()
                ready = client.get("/ready").json()
                if job["profile"] != expected or ready["loaded_profile"] != expected:
                    raise RuntimeError("Reported profile does not match requested/auto-selected profile")
                media = json.loads(subprocess.check_output(
                    ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output)], text=True
                ))
                video = next(s for s in media["streams"] if s["codec_type"] == "video")
                audio = next(s for s in media["streams"] if s["codec_type"] == "audio")
                if (video["width"], video["height"], int(video["nb_frames"]), audio["channels"]) != (864, 480, 124, 2):
                    raise RuntimeError("Unexpected output geometry or audio channels")
                subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(output),
                                "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"], check=True)
                item = {"label": label, "job": job, "wall_including_switch_seconds": elapsed,
                        "output": str(output), "loaded_profile": ready["loaded_profile"]}
                summary["results"].append(item)
                print(json.dumps(item), flush=True)
            summary["status"] = "completed"
        except BaseException as exc:
            summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
