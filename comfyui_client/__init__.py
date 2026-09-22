"""Remote-only H3 node; local config.json or process environment supplies credentials."""

import re
import tempfile
import time
import uuid
from pathlib import Path

import requests

from .config import connection_settings


def sampling_payload(sampling, steps):
    if sampling == "custom":
        if type(steps) is not int or not 1 <= steps <= 49:
            raise ValueError("H3 custom steps must be an integer from 1 to 49")
        return {"steps": steps}
    if sampling in ("fast", "quality"):
        return {"preset": sampling}
    if sampling == "server_default":
        return {}
    raise ValueError("Unknown H3 sampling mode")


def checked_id(value, prefix):
    if not isinstance(value, str) or not re.fullmatch(prefix + r"_[a-f0-9]{32}", value):
        raise RuntimeError("H3 server returned an invalid identifier")
    return value


def check_response(response):
    if 300 <= response.status_code < 400:
        raise RuntimeError("H3 API redirects are not followed; configure the final server URL")
    response.raise_for_status()


class H3RemoteVideo:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # A queued remote request is a side effect; do not cache it across runs.
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task": (["t2va", "fl2va", "ref2va"],),
                "prompt": ("STRING", {"multiline": True}),
                "seconds": ("FLOAT", {"default": 5, "min": 4, "max": 15}),
                "short_edge": ([480, 768],),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],),
                "reference_video_path": ("STRING", {"default": ""}),
                "reference_audio_path": ("STRING", {"default": ""}),
            },
            "optional": {
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "reference_images": ("IMAGE",),
                "sampling": (["server_default", "fast", "quality", "custom"],),
                "steps": ("INT", {"default": 8, "min": 1, "max": 49}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "generate"
    CATEGORY = "H3 API"
    OUTPUT_NODE = True

    def generate(
        self,
        task,
        prompt,
        seconds,
        short_edge,
        seed,
        aspect_ratio,
        reference_video_path="",
        reference_audio_path="",
        first_frame=None,
        last_frame=None,
        reference_images=None,
        sampling="server_default",
        steps=8,
    ):
        import comfy.model_management
        import folder_paths

        base, api_key = connection_settings()
        sampling_args = sampling_payload(sampling, steps)
        # Keep credentials out of serialized workflows.
        headers = {"Authorization": "Bearer " + api_key}
        conditions = []
        with requests.Session() as client, tempfile.TemporaryDirectory(prefix="h3-upload-") as temp:
            client.trust_env = False
            client.headers.update(headers)

            def upload(path, role, index=None):
                with Path(path).open("rb") as stream:
                    response = client.post(
                        base + "/v1/files", files={"file": (Path(path).name, stream)}, timeout=120,
                        allow_redirects=False,
                    )
                check_response(response)
                file = response.json()
                condition = {"file_id": checked_id(file["id"], "file"), "type": file["type"], "role": role}
                if index is not None:
                    condition["frame_index"] = index
                conditions.append(condition)

            for batch, role, frame_index in (
                (first_frame, "keyframe", 0),
                (last_frame, "keyframe", -1),
                (reference_images, "reference", None),
            ):
                if batch is None:
                    continue
                import numpy as np
                from PIL import Image

                if role == "keyframe" and len(batch) != 1:
                    raise ValueError("A keyframe input must contain exactly one image")
                for tensor in batch:
                    path = Path(temp) / f"{uuid.uuid4().hex}.png"
                    Image.fromarray((tensor.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)).save(path)
                    upload(path, role, frame_index)
            for path in (reference_video_path, reference_audio_path):
                if path.strip():
                    upload(path, "reference")
            response = client.post(
                base + "/v1/videos",
                json={
                    "task": task,
                    "prompt": prompt,
                    "conditions": conditions,
                    "seed": seed,
                    "quality": "lossless",
                    **sampling_args,
                    "target": {
                        "short_edge": int(short_edge),
                        "aspect_ratio": aspect_ratio,
                        "duration_seconds": seconds,
                    },
                },
                timeout=30,
                allow_redirects=False,
            )
            check_response(response)
            job_id = checked_id(response.json()["id"], "video")
            print(f"H3 API task: {job_id}; interrupting this node does not cancel the remote task.")
            deadline = time.monotonic() + 4 * 3600
            previous_status = None
            while time.monotonic() < deadline:
                comfy.model_management.throw_exception_if_processing_interrupted()
                response = client.get(base + f"/v1/videos/{job_id}", timeout=30, allow_redirects=False)
                check_response(response)
                job = response.json()
                if job["status"] != previous_status:
                    previous_status = job["status"]
                    print(f"H3 API {job_id}: {previous_status}")
                if job["status"] in ("failed", "cancelled", "canceled"):
                    raise RuntimeError(job.get("error") or "H3 remote task failed or was cancelled")
                if job["status"] == "completed":
                    path = Path(folder_paths.get_output_directory()) / "h3_api" / f"{job_id}.mp4"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    partial = path.with_suffix(".partial.mp4")
                    try:
                        with client.get(
                            base + f"/v1/videos/{job_id}/content", stream=True, timeout=120,
                            allow_redirects=False,
                        ) as result:
                            check_response(result)
                            content_type = result.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                            if content_type not in ("video/mp4", "application/octet-stream"):
                                raise RuntimeError("H3 content endpoint did not return a video")
                            with partial.open("wb") as stream:
                                for chunk in result.iter_content(1024 * 1024):
                                    comfy.model_management.throw_exception_if_processing_interrupted()
                                    stream.write(chunk)
                        if partial.stat().st_size == 0:
                            raise RuntimeError("H3 server returned an empty video")
                        partial.replace(path)
                    except BaseException:
                        partial.unlink(missing_ok=True)
                        raise
                    return {
                        "ui": {
                            "text": [str(path)],
                            "h3_videos": [{"filename": path.name, "subfolder": "h3_api", "type": "output"}],
                        },
                        "result": (str(path),),
                    }
                time.sleep(2)
            raise TimeoutError(f"Wait expired; query task {job_id} through the API")


NODE_CLASS_MAPPINGS = {"H3RemoteVideo": H3RemoteVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"H3RemoteVideo": "H3 Remote API Video (BF16)"}
WEB_DIRECTORY = "./web"
