"""Remote-only node. Configure H3_API_URL and H3_API_KEY in ComfyUI's environment."""

import os
import tempfile
import time
import uuid
from pathlib import Path

import requests


class H3RemoteVideo:
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
        import numpy as np
        from PIL import Image

        base = os.environ["H3_API_URL"].rstrip("/")
        # Keep credentials out of serialized workflows.
        headers = {"Authorization": "Bearer " + os.environ["H3_API_KEY"]}
        conditions = []
        with requests.Session() as client, tempfile.TemporaryDirectory(prefix="h3-upload-") as temp:
            client.trust_env = False
            client.headers.update(headers)

            def upload(path, role, index=None):
                with Path(path).open("rb") as stream:
                    response = client.post(
                        base + "/v1/files", files={"file": (Path(path).name, stream)}, timeout=120
                    )
                response.raise_for_status()
                file = response.json()
                condition = {"file_id": file["id"], "type": file["type"], "role": role}
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
                if role == "keyframe" and len(batch) != 1:
                    raise ValueError("A keyframe input must contain exactly one image")
                for tensor in batch:
                    path = Path(temp) / f"{uuid.uuid4().hex}.png"
                    Image.fromarray((tensor.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)).save(path)
                    upload(path, role, frame_index)
            for path in (reference_video_path, reference_audio_path):
                if path.strip():
                    upload(path, "reference")
            sampling_args = {}
            if sampling == "custom":
                sampling_args["steps"] = int(steps)
            elif sampling in ("fast", "quality"):
                sampling_args["preset"] = sampling
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
            )
            response.raise_for_status()
            job_id = response.json()["id"]
            print(f"H3 API task: {job_id}; interrupting this node does not cancel the remote task.")
            deadline = time.monotonic() + 4 * 3600
            while time.monotonic() < deadline:
                comfy.model_management.throw_exception_if_processing_interrupted()
                response = client.get(base + f"/v1/videos/{job_id}", timeout=30)
                response.raise_for_status()
                job = response.json()
                if job["status"] == "failed":
                    raise RuntimeError(job["error"])
                if job["status"] == "completed":
                    path = Path(folder_paths.get_output_directory()) / "h3_api" / f"{job_id}.mp4"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    partial = path.with_suffix(".partial.mp4")
                    try:
                        with client.get(
                            base + f"/v1/videos/{job_id}/content", stream=True, timeout=120
                        ) as result:
                            result.raise_for_status()
                            with partial.open("wb") as stream:
                                for chunk in result.iter_content(1024 * 1024):
                                    comfy.model_management.throw_exception_if_processing_interrupted()
                                    stream.write(chunk)
                        partial.replace(path)
                    except BaseException:
                        partial.unlink(missing_ok=True)
                        raise
                    return {"ui": {"text": [str(path)]}, "result": (str(path),)}
                time.sleep(2)
            raise TimeoutError(f"Wait expired; query task {job_id} through the API")


NODE_CLASS_MAPPINGS = {"H3RemoteVideo": H3RemoteVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"H3RemoteVideo": "H3 Remote API Video (BF16)"}
