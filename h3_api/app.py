import asyncio
import hmac
import json
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from filelock import FileLock, Timeout
from starlette.datastructures import Headers
from starlette.responses import JSONResponse

from h3_api.backend import SGLangBackend
from h3_api.media import validate_input
from h3_api.schemas import VideoRequest
from h3_api.settings import Settings
from h3_api.store import QueueFull, Store
from h3_api.worker import Worker

EXTENSIONS = {
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".mp4": "video",
    ".mov": "video",
    ".webm": "video",
    ".wav": "audio",
    ".mp3": "audio",
    ".flac": "audio",
    ".m4a": "audio",
}


class RequestGuard:
    """Reject unauthenticated/oversized bodies before multipart spooling."""

    def __init__(self, app, settings):
        self.app = app
        self.settings = settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = Headers(scope=scope)
        protected = scope["path"].startswith("/v1/") or scope["path"] == "/ready"
        if protected:
            expected = ("Bearer " + self.settings.api_key).encode()
            if not hmac.compare_digest(headers.get("authorization", "").encode(), expected):
                return await JSONResponse(
                    {"detail": "Invalid API key"}, status_code=401, headers={"WWW-Authenticate": "Bearer"}
                )(scope, receive, send)
        limit = self.settings.max_upload_bytes + 1024 * 1024 if scope["path"] == "/v1/files" else 1024 * 1024
        try:
            declared = int(headers.get("content-length", "0"))
        except ValueError:
            return await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(
                scope, receive, send
            )
        if declared > limit:
            return await JSONResponse({"detail": "Request too large"}, status_code=413)(scope, receive, send)
        received = 0

        async def bounded_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > limit:
                raise HTTPException(413, "Request too large")
            return message

        return await self.app(scope, bounded_receive, send)


def create_app(
    settings=None,
    backend_factory=SGLangBackend,
    *,
    run_worker=True,
    input_validator=validate_input,
    output_validator=None,
):
    settings = settings or Settings.from_env()
    store = Store(settings.data_dir / "jobs.sqlite3")
    auth = HTTPBearer(auto_error=False)

    async def authorized(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(auth)]):
        if not credentials or not hmac.compare_digest(
            credentials.credentials.encode(), settings.api_key.encode()
        ):
            raise HTTPException(401, "Invalid API key", headers={"WWW-Authenticate": "Bearer"})

    @asynccontextmanager
    async def lifespan(app):
        # Prevent multiple uvicorn workers/instances from each loading a GPU model.
        lock = FileLock(settings.data_dir / "scheduler.lock")
        try:
            lock.acquire(timeout=0)
        except Timeout as exc:
            raise RuntimeError("Another H3 scheduler owns this data directory; use one API worker") from exc
        backend = backend_factory(settings)
        worker = Worker(
            settings, store, backend, **({"output_validator": output_validator} if output_validator else {})
        )
        app.state.worker = worker
        store.recover()
        task = asyncio.create_task(worker.run()) if run_worker else None
        app.state.scheduler_task = task
        try:
            yield
        finally:
            if task:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            try:
                await backend.close()
            finally:
                lock.release()

    app = FastAPI(
        title="H3 BF16 API",
        version="0.2.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(RequestGuard, settings=settings)
    app.state.store = store

    @app.get("/health")
    async def health(request: Request):
        task = request.app.state.scheduler_task
        if run_worker and task is not None and task.done():
            raise HTTPException(503, "Scheduler stopped; inspect service logs")
        return {"status": "ok"}

    @app.get("/docs", include_in_schema=False, response_class=HTMLResponse)
    async def docs():
        # The UI is public so users can enter their key with Swagger's Authorize button.
        return get_swagger_ui_html(openapi_url="/openapi.json", title="H3 API")

    @app.get("/openapi.json", include_in_schema=False)
    async def openapi():
        return app.openapi()

    @app.get("/ready", dependencies=[Depends(authorized)])
    async def ready(request: Request):
        worker = request.app.state.worker
        task = request.app.state.scheduler_task
        alive = not run_worker or (task is not None and not task.done())
        return {
            "accepting_requests": alive,
            "backend_state": worker.backend.state,
            "loaded_variant": worker.backend.variant,
            "active_job_id": worker.active_id,
            "jobs": store.counts(),
            "last_error": worker.last_error,
            "sampling": {
                "mode": "turbo" if settings.turbo_lora else "base",
                "default_steps": 8 if settings.turbo_lora else 49,
                "presets": {"fast": 4, "quality": 8} if settings.turbo_lora else {},
                "custom_steps_range": [1, 49] if settings.turbo_lora else None,
                "supported_tasks": ["t2va", "fl2va"] if settings.turbo_lora else ["t2va", "fl2va", "ref2va"],
            },
        }

    @app.post("/v1/files", dependencies=[Depends(authorized)], status_code=201)
    async def upload(file: UploadFile):
        extension = Path(file.filename or "").suffix.lower()
        kind = EXTENSIONS.get(extension)
        if not kind:
            await file.close()
            raise HTTPException(415, "Unsupported file extension")
        file_id = f"file_{uuid.uuid4().hex}"
        uploads = settings.data_dir / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        final = uploads / (file_id + extension)
        partial = uploads / (file_id + ".partial")
        size = 0
        try:
            with partial.open("wb") as stream:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise HTTPException(413, "File exceeds 100 MiB")
                    stream.write(chunk)
            if not size:
                raise HTTPException(422, "Empty upload")
            partial.replace(final)
            try:
                await input_validator(final, kind)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            store.add_file(file_id, kind, final, size)
        except BaseException:
            partial.unlink(missing_ok=True)
            final.unlink(missing_ok=True)
            raise
        finally:
            await file.close()
        return {"id": file_id, "type": kind, "bytes": size}

    def public(job):
        saved = json.loads(job["request"])
        return {
            "id": job["id"],
            "status": job["status"],
            "created_at": job["created"],
            "updated_at": job["updated"],
            "error": job["error"],
            "sampling": saved.get("_sampling"),
            "content_url": f"/v1/videos/{job['id']}/content" if job["status"] == "completed" else None,
        }

    @app.post("/v1/videos", dependencies=[Depends(authorized)], status_code=202)
    async def submit(body: VideoRequest, request: Request):
        task = request.app.state.scheduler_task
        if run_worker and (task is None or task.done()):
            raise HTTPException(503, "Scheduler unavailable")
        try:
            sampling = body.sampling(settings.turbo_lora is not None)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        for condition in body.conditions:
            file = store.file(condition.file_id)
            if not file or not Path(file["path"]).is_file():
                raise HTTPException(422, f"Unknown or missing file: {condition.file_id}")
            if file["kind"] != condition.type:
                raise HTTPException(422, f"File type mismatch: {condition.file_id}")
        try:
            job_id = store.submit({**body.model_dump(), "_sampling": sampling}, settings.max_queue)
        except QueueFull as exc:
            raise HTTPException(429, "Queue full; retry later", headers={"Retry-After": "60"}) from exc
        return public(store.get(job_id))

    @app.get("/v1/videos/{job_id}", dependencies=[Depends(authorized)])
    async def status(job_id: str):
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, "Unknown task")
        return public(job)

    @app.get("/v1/videos/{job_id}/content", dependencies=[Depends(authorized)])
    async def content(job_id: str):
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, "Unknown task")
        if job["status"] != "completed":
            raise HTTPException(409, "Task has not completed")
        path = Path(job["output"])
        if not path.is_file():
            raise HTTPException(410, "Output no longer available")
        return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")

    return app
