import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from h3_api.app import create_app
from h3_api.schemas import VideoRequest
from h3_api.settings import Settings
from h3_api.store import Store
from h3_api.worker import Worker

KEY = "test-only-key-not-for-deployment"
HEADERS = {"Authorization": f"Bearer {KEY}"}


class FakeBackend:
    def __init__(self, settings):
        self.state = "idle"
        self.variant = None
        self.events = []

    async def ensure(self, variant):
        if variant != self.variant:
            await self.stop()
            self.events.append(("load", variant))
            self.variant = variant
        self.state = "ready"

    async def generate(self, payload, destination):
        self.events.append(("generate", self.variant))
        destination.write_bytes(b"fake-mp4")

    async def stop(self):
        self.events.append(("stop", self.variant))
        self.variant = None
        self.state = "idle"

    async def close(self):
        await self.stop()


async def accept_media(*args):
    pass


def config(tmp_path):
    return Settings(tmp_path / "data", tmp_path / "models", KEY, poll_interval=0.001)


def app(tmp_path, **kwargs):
    return create_app(
        config(tmp_path), FakeBackend, input_validator=accept_media, output_validator=accept_media, **kwargs
    )


def test_auth_upload_validation_and_queue(tmp_path):
    service = app(tmp_path, run_worker=False)
    with TestClient(service) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 401
        assert client.post("/v1/videos", json={"task": "t2va", "prompt": "hello"}).status_code == 401
        client.headers.update(HEADERS)
        assert client.post("/v1/files", files={"file": ("x.exe", b"bad")}).status_code == 415
        uploaded = client.post("/v1/files", files={"file": ("../../photo.png", b"png")}).json()
        assert (config(tmp_path).data_dir / "uploads" / (uploaded["id"] + ".png")).exists()
        body = {
            "task": "fl2va",
            "prompt": "animate",
            "conditions": [
                {"file_id": uploaded["id"], "type": "image", "role": "keyframe", "frame_index": 0}
            ],
        }
        response = client.post("/v1/videos", json=body)
        assert response.status_code == 202
        job = response.json()
        assert client.get(f"/v1/videos/{job['id']}/content").status_code == 409
        assert client.get("/v1/videos/missing").status_code == 404
        body["conditions"][0]["file_id"] = "file_" + "0" * 32
        assert client.post("/v1/videos", json=body).status_code == 422
        for _ in range(31):
            assert client.post("/v1/videos", json={"task": "t2va", "prompt": "test"}).status_code == 202
        assert client.post("/v1/videos", json={"task": "t2va", "prompt": "test"}).status_code == 429


@pytest.mark.parametrize(
    "changes",
    [
        {"quality": "high"},
        {"task": "unknown"},
        {"prompt": " "},
        {"seed": -1},
        {"task": "fl2va"},
        {"task": "ref2va"},
        {"target": {"short_edge": 2000}},
        {"target": {"duration_seconds": 30}},
        {"target": {"aspect_ratio": "auto"}},
        {"quantization": "fp8"},
    ],
)
def test_reject_invalid_or_approximate_requests(changes):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        VideoRequest.model_validate({"task": "t2va", "prompt": "test", **changes})


def test_fifo_switching_reuses_worker_and_serves_content(tmp_path):
    service = app(tmp_path, run_worker=False)
    with TestClient(service) as client:
        client.headers.update(HEADERS)
        uploaded = client.post("/v1/files", files={"file": ("photo.png", b"png")}).json()
        ids = []
        for kind in ("t2va", "t2va", "ref2va", "t2va"):
            body = {"task": kind, "prompt": "test"}
            if kind == "ref2va":
                body["conditions"] = [{"type": "image", "file_id": uploaded["id"], "role": "reference"}]
            ids.append(client.post("/v1/videos", json=body).json()["id"])
        worker = service.state.worker
        for job_id in ids:
            assert asyncio.run(worker.run_one())
            assert client.get(f"/v1/videos/{job_id}").json()["status"] == "completed"
            assert client.get(f"/v1/videos/{job_id}/content").content == b"fake-mp4"
        assert [event for event in worker.backend.events if event[0] == "load"] == [
            ("load", "fl2va"),
            ("load", "ref2va"),
            ("load", "fl2va"),
        ]
        assert not asyncio.run(worker.run_one())


def test_restart_marks_inflight_failed_preserves_queued_and_completed(tmp_path):
    settings = config(tmp_path)
    store = Store(settings.data_dir / "jobs.sqlite3")
    ids = [store.submit({"task": "t2va", "prompt": "x"}, 32) for _ in range(4)]
    store.update(ids[0], "running")
    store.update(ids[1], "loading")
    store.update(ids[2], "completed", output="result.mp4")
    store.recover()
    assert [store.get(i)["status"] for i in ids] == ["failed", "failed", "completed", "queued"]


@pytest.mark.parametrize("failure", ["timeout", "bad_output", "backend_error"])
def test_failure_stops_backend_and_next_job_can_run(tmp_path, failure):
    settings = config(tmp_path)
    object.__setattr__(settings, "task_timeout", 0.02)
    store = Store(settings.data_dir / "jobs.sqlite3")
    backend = FakeBackend(settings)
    original = backend.generate

    async def broken(payload, destination):
        destination.write_bytes(b"partial")
        if failure == "timeout":
            await asyncio.sleep(5)
        elif failure == "backend_error":
            raise RuntimeError("backend failed")

    async def bad_output(path):
        raise ValueError("no stereo audio")

    backend.generate = broken
    worker = Worker(settings, store, backend, bad_output if failure == "bad_output" else accept_media)
    first = store.submit({"task": "t2va", "prompt": "x"}, 32)
    second = store.submit({"task": "t2va", "prompt": "y"}, 32)
    asyncio.run(worker.run_one())
    assert store.get(first)["status"] == "failed"
    assert backend.variant is None
    assert not list((settings.data_dir / "outputs").glob("*.partial.mp4"))
    backend.generate = original
    worker.output_validator = accept_media
    asyncio.run(worker.run_one())
    assert store.get(second)["status"] == "completed"


def test_only_one_scheduler_per_directory(tmp_path):
    with (
        TestClient(app(tmp_path, run_worker=False)),
        pytest.raises(RuntimeError, match="Another H3 scheduler"),
        TestClient(app(tmp_path, run_worker=False)),
    ):
        pass


def test_payload_uses_local_uploaded_path_not_client_path(tmp_path):
    settings = config(tmp_path)
    store = Store(settings.data_dir / "jobs.sqlite3")
    source = tmp_path / "input.png"
    source.write_bytes(b"png")
    file_id = "file_" + "1" * 32
    store.add_file(file_id, "image", source, 3)
    request = VideoRequest(
        task="fl2va",
        prompt="x",
        conditions=[{"type": "image", "file_id": file_id, "role": "keyframe", "frame_index": -1}],
    )
    payload = Worker(settings, store, FakeBackend(settings)).payload(request)
    assert payload["conditions"] == [
        {"type": "image", "uri": str(source), "role": "keyframe", "frame_index": -1}
    ]
    assert payload["quality"] == "lossless"


@pytest.mark.parametrize("variant,partition", [("fl2va", "FL2VA"), ("ref2va", "Ref2VA")])
@pytest.mark.parametrize("use_turbo", [False, True])
def test_backend_command_uses_all_four_local_components(tmp_path, variant, partition, use_turbo):
    from dataclasses import replace

    from h3_api.backend import SGLangBackend

    settings = config(tmp_path)
    if use_turbo:
        lora = tmp_path / "lora.safetensors"
        lora.touch()
        settings = replace(settings, turbo_lora=lora)
    root = settings.models_dir / "prepared"
    root.mkdir(parents=True)
    config_root = root / "config"
    config_root.mkdir()
    # The official modular root is intentionally NOT a native pipeline directory.
    (config_root / "model_index.json").write_text(json.dumps({"_class_name": "MiniMaxH3ModularPipeline"}))
    selected = config_root / partition
    selected.mkdir()
    for name in ("transformer", "text_encoder", "video_vae", "audio_vae", "processor", "tokenizer"):
        (selected / name).mkdir()
    index = {
        "_class_name": "MiniMaxH3Pipeline",
        "_diffusers_version": "0.32.2",
        "_minimax_h3": {"partition": variant},
    }
    (selected / "model_index.json").write_text(json.dumps(index))
    components = {}
    for component in ("transformer", "text_encoder", "video_vae", "audio_vae"):
        path = root / f"{component}.safetensors"
        path.touch()
        components[component] = str(path)
    (root / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "config_root": str(config_root), "variants": {variant: components}})
    )
    backend = SGLangBackend(settings)
    if use_turbo and variant == "ref2va":
        with pytest.raises(ValueError, match="cannot be applied"):
            backend.command(variant)
        asyncio.run(backend.close())
        return
    args = backend.command(variant)
    if use_turbo:
        assert args[args.index("--lora-path") + 1] == str(lora.parent)
        assert args[args.index("--lora-weight-name") + 1] == lora.name
        assert args[args.index("--lora-merge-mode") + 1] == "dynamic"
        assert args[args.index("--warmup-mode") + 1] == "off"
    assert args[args.index("--model-path") + 1] == str(selected)
    assert args[args.index("--model-subfolder") + 1] == "."
    assert args[args.index("--input-save-path") + 1] == str(settings.data_dir / "sglang" / "inputs")
    assert args[args.index("--output-path") + 1] == str(settings.data_dir / "sglang" / "outputs")
    assert "--model-variant" not in args  # Avoid a second partition suffix in the loader.
    for component, path in components.items():
        assert args[args.index(f"--component-weights-paths.{component}") + 1] == path
    assert "--quantization" not in args
    assert args[args.index("--host") + 1] == "127.0.0.1"
    index["_minimax_h3"]["partition"] = "ref2va" if variant == "fl2va" else "fl2va"
    (selected / "model_index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="Invalid native"):
        backend.command(variant)
    asyncio.run(backend.close())


def test_backend_working_directory_and_spill_use_data_mount(tmp_path, monkeypatch):
    from pathlib import Path

    from h3_api.backend import SGLangBackend

    for name in ("H3_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.setenv(name, "test-secret")
    monkeypatch.setenv("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt")
    settings = config(tmp_path)
    backend = SGLangBackend(settings)
    context = backend.launch_context()
    assert context["cwd"] == str(settings.data_dir / "sglang")
    env = context["env"]
    assert all(name not in env for name in ("H3_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"))
    assert env["SSL_CERT_FILE"] == "/etc/ssl/certs/ca-certificates.crt"
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    for name, directory in (
        ("HF_HOME", "huggingface"),
        ("SGLANG_DIFFUSION_CACHE_ROOT", "sglang"),
        ("SGLANG_DIFFUSION_HOST_SPILL_DIR", "host_spill"),
    ):
        assert Path(env[name]) == settings.data_dir / "cache" / directory
        assert Path(env[name]).is_dir()
    for directory in ("inputs", "outputs"):
        assert (Path(context["cwd"]) / directory).is_dir()
    assert backend.launch_context()["cwd"] == context["cwd"]  # Reused after model switch.
    asyncio.run(backend.close())
