import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from h3_api import ref2va, turbo
from h3_api.app import create_app
from h3_api.backend import SGLangBackend
from h3_api.profile_backend import ProfileBackend
from h3_api.profiles import LARRY, LIGHTX, resolve
from h3_api.schemas import VideoRequest
from h3_api.settings import Settings
from production import install_ref2va

KEY = "profiles-test-only-not-a-deployment-key"


def settings(tmp_path):
    return Settings(tmp_path / "data", tmp_path / "models", KEY,
                    turbo_lora=tmp_path / "larry" / "adapter",
                    ref2va_lora=tmp_path / "lightx" / "adapter")


class FakeBackend:
    def __init__(self, settings):
        self.state = "idle"
        self.variant = self.loaded_profile = None
        self.events, self.payloads = [], []

    async def ensure(self, variant, profile=None):
        if (variant, profile) != (self.variant, self.loaded_profile):
            await self.stop()
            self.events.append(("load", variant, profile))
        self.variant, self.loaded_profile, self.state = variant, profile, "ready"

    async def generate(self, payload, destination):
        self.payloads.append(payload)
        destination.write_bytes(b"fixture-video")

    async def stop(self):
        self.events.append(("stop", self.variant, self.loaded_profile))
        self.variant = self.loaded_profile = None
        self.state = "idle"

    async def close(self):
        await self.stop()


async def accept(*args):
    pass


def ref_request(**kwargs):
    return {"task": "ref2va", "prompt": "reference", "conditions": [
        {"type": "image", "role": "reference", "file_id": "file_" + "a" * 32}], **kwargs}


@pytest.mark.parametrize("fields", [{"preset": "quality"}, {"steps": 8}, {"steps": 12},
                                   {"profile": LARRY}, {"profile": "/tmp/custom.safetensors"}])
def test_reject_wrong_ref_recipe(tmp_path, fields):
    with pytest.raises(ValueError):
        resolve(VideoRequest.model_validate(ref_request(**fields)), settings(tmp_path))


def test_profile_selection_and_restrictions(tmp_path):
    config = settings(tmp_path)
    for fields in ({}, {"preset": "fast"}, {"steps": 4}, {"profile": LIGHTX}):
        name, sampling = resolve(VideoRequest.model_validate(ref_request(**fields)), config)
        assert (name, sampling["steps"], sampling["num_inference_steps"]) == (LIGHTX, 4, 5)
    with pytest.raises(ValueError, match="not enabled"):
        resolve(VideoRequest.model_validate(ref_request(profile=LIGHTX)), replace(config, ref2va_lora=None))
    with pytest.raises(ValueError, match="task=ref2va"):
        resolve(VideoRequest(task="t2va", prompt="x", profile=LIGHTX), config)
    body = ref_request()
    body["conditions"][0]["type"] = "video"
    with pytest.raises(ValueError, match="images only"):
        resolve(VideoRequest.model_validate(body), config)


def test_queue_profile_persistence_switches_and_old_job_access(tmp_path):
    service = create_app(settings(tmp_path), FakeBackend, run_worker=False,
                         input_validator=accept, output_validator=accept)
    with TestClient(service) as client:
        assert client.get("/v1/profiles").status_code == 401
        client.headers.update({"Authorization": f"Bearer {KEY}"})
        assert [p["id"] for p in client.get("/v1/profiles").json()["profiles"]] == [LARRY, LIGHTX]
        file = client.post("/v1/files", files={"file": ("ref.png", b"fixture")}).json()["id"]
        reference = ref_request()
        reference["conditions"][0]["file_id"] = file
        bodies = [{"task": "t2va", "prompt": "x", "preset": "fast"}, reference,
                  {**reference, "profile": LIGHTX, "steps": 4},
                  {"task": "t2va", "prompt": "x", "steps": 8, "profile": LARRY}]
        for body, expected in zip(bodies, (LARRY, LIGHTX, LIGHTX, LARRY)):
            reply = client.post("/v1/videos", json=body)
            assert reply.status_code == 202, reply.text
            job = reply.json()
            assert job["profile"] == expected
            assert asyncio.run(service.state.worker.run_one())
            assert client.get(f"/v1/videos/{job['id']}").json()["status"] == "completed"
            assert client.get(f"/v1/videos/{job['id']}/content").content == b"fixture-video"
        backend = service.state.worker.backend
        assert [e[2] for e in backend.events if e[0] == "load"] == [LARRY, LIGHTX, LARRY]
        assert backend.payloads[1]["flow_shift"] == 12
        assert backend.payloads[1]["audio_flow_shift"] == 3
        assert "flow_shift" not in backend.payloads[3]
        assert all("profile" not in p for p in backend.payloads)
        before = service.state.store.counts()
        assert client.post("/v1/videos", json={**reference, "steps": 12}).status_code == 422
        assert service.state.store.counts() == before
        schema = client.get("/openapi.json").json()
        assert "JobResponse" in schema["components"]["schemas"]
        assert client.get("/ready").json()["loaded_profile"] == LARRY


def test_real_profile_backend_reuse_switch_order_and_failure_cleanup(tmp_path, monkeypatch):
    config = settings(tmp_path)
    events = []

    async def stop(self):
        events.append(("stop", self.variant))
        self.process = self.variant = None
        self.state = "idle"

    async def ensure(self, variant):
        await self.stop()
        events.append(("spawn", variant))
        self.variant, self.state, self.process = variant, "ready", SimpleNamespace(returncode=None)
        log = config.data_dir / "logs" / f"sglang-{variant}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        asset = config.ref2va_lora if variant == "ref2va" else config.turbo_lora
        count = 208 if variant == "ref2va" else 259
        with log.open("a") as stream:
            stream.write(f"LoRA adapter(s) {asset.parent} applied to {count} layers (targets: all, strengths: 1.00, merge_mode=dynamic)\n")

    monkeypatch.setattr(SGLangBackend, "stop", stop)
    monkeypatch.setattr(SGLangBackend, "ensure", ensure)
    monkeypatch.setattr(turbo, "verify_lora", lambda path: events.append(("hash", LARRY)))
    monkeypatch.setattr(ref2va, "verify_lora", lambda path: events.append(("hash", LIGHTX)))
    backend = ProfileBackend(config)

    async def scenario():
        await backend.ensure("fl2va", LARRY)
        count = len(events)
        await backend.ensure("fl2va", LARRY)
        assert len(events) == count  # Changing steps cannot affect model identity.
        await backend.ensure("ref2va", LIGHTX)
        assert backend.loaded_profile == LIGHTX
        await backend.ensure("fl2va", LARRY)
        assert backend.loaded_profile == LARRY
        with pytest.raises(ValueError):
            await backend.ensure("fl2va", LIGHTX)
        assert backend.loaded_profile == LARRY
        # Stale successful log lines must not hide an activation failure on a new load.
        monkeypatch.setattr(ref2va, "verify_active", lambda *a: (_ for _ in ()).throw(RuntimeError("bad activation")))
        with pytest.raises(RuntimeError, match="activation"):
            await backend.ensure("ref2va", LIGHTX)
        assert backend.loaded_profile is None and backend.process is None
        await backend.close()

    asyncio.run(scenario())
    hashes = [i for i, e in enumerate(events) if e[0] == "hash"]
    assert all(events[i - 1][0] == "stop" for i in hashes)


def test_shutdown_failure_blocks_new_load(tmp_path, monkeypatch):
    backend = ProfileBackend(settings(tmp_path))

    async def broken_stop(self):
        raise RuntimeError("Backend children still alive")

    monkeypatch.setattr(SGLangBackend, "stop", broken_stop)
    monkeypatch.setattr(ref2va, "verify_lora", lambda _: pytest.fail("No new load before shutdown"))
    with pytest.raises(RuntimeError, match="children"):
        asyncio.run(backend.ensure("ref2va", LIGHTX))
    asyncio.run(backend.client.aclose())


def test_preprocessing_flag_and_cache_do_not_leak_to_larry(tmp_path, monkeypatch):
    backend = ProfileBackend(settings(tmp_path))
    monkeypatch.setattr(backend, "validate_comfy_qkv_contract", lambda _: None)
    monkeypatch.setenv("H3_REF2VA_REFERENCE_POLICY", "inherited-untrusted")
    backend.selected_profile = LIGHTX
    lightx = backend.launch_context("ref2va")["env"]
    backend.selected_profile = LARRY
    larry = backend.launch_context("fl2va")["env"]
    assert lightx["H3_REF2VA_REFERENCE_POLICY"] == "lightx2v-match-v1"
    assert "H3_REF2VA_REFERENCE_POLICY" not in larry
    assert larry["H3_COMFY_BF16_QKV_LAYOUT"] == "concatenated"
    assert lightx["SGLANG_DIFFUSION_HOST_SPILL_DIR"] != larry["SGLANG_DIFFUSION_HOST_SPILL_DIR"]
    asyncio.run(backend.close())


def test_adapter_install_retains_source_and_refuses_overwrite(tmp_path, monkeypatch):
    def verify(path):
        if path.read_bytes() != b"pinned":
            raise ValueError("Wrong SHA")

    monkeypatch.setattr(install_ref2va, "verify_lora", verify)
    source, dest = tmp_path / "source", tmp_path / "installed"
    source.write_bytes(b"pinned")
    install_ref2va.install(source, dest)
    assert source.read_bytes() == dest.read_bytes() == b"pinned"
    dest.write_bytes(b"unrelated")
    with pytest.raises(ValueError):
        install_ref2va.install(source, dest)
    assert dest.read_bytes() == b"unrelated"
    assert not list(tmp_path.glob(".ref2va-*"))
