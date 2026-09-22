import asyncio
import json
import logging

from h3_api.media import validate_output
from h3_api.schemas import VideoRequest

logger = logging.getLogger(__name__)


class Worker:
    def __init__(self, settings, store, backend, output_validator=validate_output):
        self.settings = settings
        self.store = store
        self.backend = backend
        self.output_validator = output_validator
        self.active_id = None
        self.last_error = None

    def payload(self, request):
        data = request.model_dump()
        data.pop("preset")
        data.pop("steps")
        data["num_inference_steps"] = request.sampling(self.settings.turbo_lora is not None)[
            "num_inference_steps"
        ]
        for condition in data["conditions"]:
            uploaded = self.store.file(condition.pop("file_id"))
            if not uploaded:
                raise ValueError("input_file_missing")
            from pathlib import Path

            if not Path(uploaded["path"]).is_file():
                raise ValueError("input_file_missing")
            condition["uri"] = uploaded["path"]
            if condition["frame_index"] is None:
                del condition["frame_index"]
        return data

    async def run_one(self):
        job = self.store.claim()
        if not job:
            return False
        job_id = self.active_id = job["id"]
        outputs = self.settings.data_dir / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        partial = outputs / f"{job_id}.partial.mp4"
        final = outputs / f"{job_id}.mp4"
        phase = "loading"
        try:
            saved = json.loads(job["request"])
            sampling = saved.pop("_sampling", None)
            request = VideoRequest.model_validate(saved)
            if sampling is not None and sampling != request.sampling(self.settings.turbo_lora is not None):
                raise ValueError("Service sampling mode changed since submission; resubmit this task")
            payload = self.payload(request)
            await self.backend.ensure(request.variant)
            phase = "running"
            self.store.update(job_id, phase)
            async with asyncio.timeout(self.settings.task_timeout):
                await self.backend.generate(payload, partial)
                await self.output_validator(partial)
            partial.replace(final)
            self.store.update(job_id, "completed", output=str(final))
            self.last_error = None
        except asyncio.CancelledError:
            self.store.update(job_id, "failed", error=f"{phase}: service_interrupted")
            raise
        except Exception as exc:
            self.last_error = f"{phase}: {type(exc).__name__}: {exc}"[:1500]
            logger.exception("H3 job %s failed during %s", job_id, phase)
            self.store.update(job_id, "failed", error=self.last_error)
            # A timeout or bad output must not leave a backend task consuming the GPU.
            await self.backend.stop()
        finally:
            partial.unlink(missing_ok=True)
            self.active_id = None
        return True

    async def run(self):
        while True:
            if not await self.run_one():
                await asyncio.sleep(0.5)
