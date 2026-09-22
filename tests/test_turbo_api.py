import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from benchmark.assets import ASSETS
from h3_api import turbo
from h3_api.app import create_app
from h3_api.backend import SGLangBackend
from h3_api.schemas import VideoRequest
from h3_api.settings import Settings
from h3_api.store import Store
from h3_api.worker import Worker

KEY = "turbo-tests-only-never-deploy-this-key"


def settings(tmp_path):
    return Settings(tmp_path / "data", tmp_path / "models", KEY, turbo_lora=tmp_path / "lora.safetensors")


class FakeBackend:
    def __init__(self, settings):
        self.state = "idle"
        self.variant = None
        self.loads = 0
        self.payloads = []

    async def ensure(self, variant):
        if variant != self.variant:
            self.loads += 1
        self.variant, self.state = variant, "ready"

    async def generate(self, payload, path):
        self.payloads.append(payload)
        path.write_bytes(b"test-video")

    async def stop(self):
        self.state, self.variant = "idle", None

    async def close(self):
        await self.stop()


async def accept(*args):
    pass


@pytest.mark.parametrize(
    "fields,nfe",
    [
        ({}, 8),
        ({"preset": "fast"}, 4),
        ({"preset": "quality"}, 8),
        ({"steps": 12}, 12),
        ({"steps": 1}, 1),
        ({"steps": 49}, 49),
    ],
)
def test_sampling_and_wire_payload(tmp_path, fields, nfe):
    config = settings(tmp_path)
    request = VideoRequest(task="t2va", prompt="car", **fields)
    store = Store(tmp_path / "jobs.db")
    payload = Worker(config, store, FakeBackend(config)).payload(request)
    assert payload["num_inference_steps"] == nfe + 1
    assert "preset" not in payload and "steps" not in payload
    assert payload["quality"] == "lossless"


@pytest.mark.parametrize(
    "fields",
    [
        {"steps": 0},
        {"steps": 50},
        {"steps": True},
        {"steps": 4.5},
        {"steps": "12"},
        {"preset": "fast", "steps": 4},
        {"preset": "other"},
        {"num_inference_steps": 13},
        {"_sampling": {"mode": "base"}},
    ],
)
def test_sampling_rejects_ambiguous_or_invalid_input(fields):
    with pytest.raises(ValidationError):
        VideoRequest(task="t2va", prompt="car", **fields)


def test_turbo_defaults_presets_and_custom_steps_share_one_worker(tmp_path):
    app = create_app(
        settings(tmp_path), FakeBackend, run_worker=False, input_validator=accept, output_validator=accept
    )
    with TestClient(app) as client:
        client.headers.update({"Authorization": f"Bearer {KEY}"})
        ready = client.get("/ready").json()
        assert ready["sampling"]["default_steps"] == 8
        for fields, nfe in (({}, 8), ({"preset": "fast"}, 4), ({"steps": 12}, 12)):
            response = client.post("/v1/videos", json={"task": "t2va", "prompt": "car", **fields})
            assert response.status_code == 202
            body = response.json()
            assert body["sampling"] == {"mode": "turbo", "steps": nfe, "num_inference_steps": nfe + 1}
            assert asyncio.run(app.state.worker.run_one())
            status = client.get(f"/v1/videos/{body['id']}").json()
            assert status["status"] == "completed"
            assert status["sampling"] == body["sampling"]
        backend = app.state.worker.backend
        assert backend.loads == 1
        assert [p["num_inference_steps"] for p in backend.payloads] == [9, 5, 13]
        upload = client.post("/v1/files", files={"file": ("ref.png", b"png")}).json()
        response = client.post(
            "/v1/videos",
            json={
                "task": "ref2va",
                "prompt": "car",
                "conditions": [{"type": "image", "file_id": upload["id"], "role": "reference"}],
            },
        )
        assert response.status_code == 422
        assert "separate recipe" in response.text


def test_plain_service_retains_baseline_and_rejects_turbo_settings(tmp_path):
    config = replace(settings(tmp_path), turbo_lora=None)
    app = create_app(config, FakeBackend, run_worker=False)
    with TestClient(app) as client:
        client.headers.update({"Authorization": f"Bearer {KEY}"})
        assert (
            client.post("/v1/videos", json={"task": "t2va", "prompt": "car", "steps": 12}).status_code == 422
        )
        assert (
            client.post("/v1/videos", json={"task": "t2va", "prompt": "car"}).json()["sampling"]["steps"]
            == 49
        )


def test_sampling_configuration_change_does_not_silently_reinterpret_queue(tmp_path):
    config = replace(settings(tmp_path), turbo_lora=None)
    store = Store(tmp_path / "jobs.db")
    job = store.submit(
        {
            "task": "t2va",
            "prompt": "car",
            "_sampling": {
                "mode": "turbo",
                "steps": 8,
                "num_inference_steps": 9,
            },
        },
        32,
    )
    backend = FakeBackend(config)
    asyncio.run(Worker(config, store, backend, accept).run_one())
    assert "mode changed" in store.get(job)["error"]
    assert backend.loads == 0


def test_actual_backend_reuses_loaded_model_without_rehash_or_reload(tmp_path):
    backend = SGLangBackend(settings(tmp_path))
    backend.variant, backend.state = "fl2va", "ready"
    backend.process = SimpleNamespace(returncode=None)
    try:
        asyncio.run(backend.ensure("fl2va"))  # LoRA path needn't exist: no reload or file access.
    finally:
        backend.process = None
        asyncio.run(backend.close())


def test_turbo_hash_and_activation_match_benchmark(tmp_path, monkeypatch):
    assert turbo.LORA_SHA256 == ASSETS["turbo"]["sha256"]
    assert turbo.LORA_BYTES == ASSETS["turbo"]["bytes"]
    path = Path("/models/loras") / turbo.LORA_FILENAME
    log = f"LoRA adapter(s) {path.parent} applied to 259 layers (targets: all, strengths: 1.00, merge_mode=dynamic)"
    assert turbo.verify_activation(log, path) == 259
    with pytest.raises(RuntimeError):
        turbo.verify_activation(log.replace("259", "0"), path)
    file = tmp_path / "lora"
    file.write_bytes(b"bad")
    with pytest.raises(ValueError, match="size"):
        turbo.verify_lora(file)
    monkeypatch.setattr(turbo, "LORA_BYTES", 3)
    with pytest.raises(ValueError, match="SHA256"):
        turbo.verify_lora(file)
