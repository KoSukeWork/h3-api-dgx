"""python examples/client.py --url http://DGX-IP:8000 --prompt '...'

Set H3_API_KEY in the client's environment. No GPU packages required.
"""

import argparse
import os
import time
from pathlib import Path

import httpx


def generate(client, payload, output: Path, timeout=14400, progress=print):
    response = client.post("/v1/videos", json=payload)
    response.raise_for_status()
    job_id = response.json()["id"]
    progress(f"Task: {job_id}")
    deadline = time.monotonic() + timeout
    previous = None
    while time.monotonic() < deadline:
        response = client.get(f"/v1/videos/{job_id}")
        response.raise_for_status()
        body = response.json()
        if body["status"] != previous:
            progress(body["status"])
            previous = body["status"]
        if body["status"] == "failed":
            raise RuntimeError(body["error"])
        if body["status"] == "completed":
            partial = output.with_suffix(".partial.mp4")
            output.parent.mkdir(parents=True, exist_ok=True)
            try:
                with client.stream("GET", f"/v1/videos/{job_id}/content") as result:
                    result.raise_for_status()
                    with partial.open("wb") as stream:
                        for chunk in result.iter_bytes():
                            stream.write(chunk)
                partial.replace(output)
            except BaseException:
                partial.unlink(missing_ok=True)
                raise
            return job_id
        time.sleep(2)
    raise TimeoutError(f"Client wait expired. Task {job_id} remains queryable and may still run.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--task", choices=["t2va", "fl2va", "ref2va"], default="t2va")
    parser.add_argument("--first-frame", type=Path)
    parser.add_argument("--last-frame", type=Path)
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument("--short-edge", type=int, choices=[480, 768], default=480)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--seed", type=int, default=42)
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--preset", choices=["fast", "quality"])
    sampling.add_argument("--steps", type=int, choices=range(1, 50), metavar="1..49")
    parser.add_argument("--output", type=Path, default=Path("h3-output.mp4"))
    args = parser.parse_args()
    with httpx.Client(
        base_url=args.url.rstrip("/"),
        timeout=120,
        trust_env=False,
        headers={"Authorization": f"Bearer {os.environ['H3_API_KEY']}"},
    ) as client:
        conditions = []
        items = [(args.first_frame, "keyframe", 0), (args.last_frame, "keyframe", -1)]
        items += [(path, "reference", None) for path in args.reference]
        for path, role, index in items:
            if not path:
                continue
            with path.open("rb") as stream:
                response = client.post("/v1/files", files={"file": (path.name, stream)})
                response.raise_for_status()
                uploaded = response.json()
            condition = {"type": uploaded["type"], "file_id": uploaded["id"], "role": role}
            if index is not None:
                condition["frame_index"] = index
            conditions.append(condition)
        payload = {
            "task": args.task,
            "prompt": args.prompt,
            "conditions": conditions,
            "target": {
                "short_edge": args.short_edge,
                "aspect_ratio": "16:9",
                "duration_seconds": args.seconds,
            },
            "seed": args.seed,
            "quality": "lossless",
        }
        if args.preset is not None:
            payload["preset"] = args.preset
        if args.steps is not None:
            payload["steps"] = args.steps
        generate(client, payload, args.output)
        print(args.output.resolve())


if __name__ == "__main__":
    main()
