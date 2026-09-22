import json
import math
import sys
import types
import zipfile

import pytest
import requests

import comfyui_client as node
from comfyui_client.config import connection_settings
from scripts.package_comfyui import CLIENT_FILES, build_archive


def write_config(path, **overrides):
    data = {"H3_API_URL": "http://dgx.example:8000/", "H3_API_KEY": "test-only-key"}
    data.update(overrides)
    path.write_text(json.dumps(data), encoding="utf-8-sig")


def test_configuration_sources_and_reload(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setenv("H3_API_URL", "https://env.example")
    monkeypatch.setenv("H3_API_KEY", "env-key")
    assert connection_settings(path) == ("https://env.example", "env-key")
    write_config(path)
    assert connection_settings(path) == ("http://dgx.example:8000", "test-only-key")
    write_config(path, H3_API_KEY="changed-key")
    assert connection_settings(path)[1] == "changed-key"
    write_config(path, H3_API_KEY=None)
    with pytest.raises(ValueError, match="Set both"):
        connection_settings(path)


@pytest.mark.parametrize("value", ["[", "[]", '{"secret": "private-value",'])
def test_bad_json_does_not_echo_content(tmp_path, value):
    path = tmp_path / "config.json"
    path.write_text(value)
    with pytest.raises(ValueError) as error:
        connection_settings(path)
    assert "private-value" not in str(error.value)


@pytest.mark.parametrize("url", [
    "http://<DGX_HOST>:8000", "ftp://dgx.example", "https://user:key@dgx.example",
    "https://dgx.example/docs", "https://dgx.example?key=secret", "http://dgx.example:bad",
    "https://dgx.example/#fragment", "http://dgx.example\\foo", "http://bad host",
])
def test_invalid_url(tmp_path, url):
    path = tmp_path / "config.json"
    write_config(path, H3_API_URL=url)
    with pytest.raises(ValueError, match="H3_API_URL"):
        connection_settings(path)


@pytest.mark.parametrize("key", ["", "REPLACE_WITH_YOUR_API_KEY", "Bearer secret", "a\nb", "a b", "密钥"])
def test_invalid_key(tmp_path, key):
    path = tmp_path / "config.json"
    write_config(path, H3_API_KEY=key)
    with pytest.raises(ValueError, match="H3_API_KEY"):
        connection_settings(path)


def test_sampling_and_workflow_contract():
    assert node.sampling_payload("fast", 12) == {"preset": "fast"}
    assert node.sampling_payload("quality", 12) == {"preset": "quality"}
    assert node.sampling_payload("custom", 12) == {"steps": 12}
    assert node.sampling_payload("server_default", 12) == {}
    for steps in (0, 50, 1.5, True):
        with pytest.raises(ValueError):
            node.sampling_payload("custom", steps)
    with pytest.raises(ValueError):
        node.sampling_payload("unknown", 8)
    assert math.isnan(node.H3RemoteVideo.IS_CHANGED())
    assert "API_KEY" not in str(node.H3RemoteVideo.INPUT_TYPES())
    assert "API_URL" not in str(node.H3RemoteVideo.INPUT_TYPES())
    with pytest.raises(RuntimeError):
        node.checked_id("../../escape", "video")


class Response:
    def __init__(self, body=None, status=200, content_type="video/mp4", chunks=(b"test-mp4",)):
        self.body = body
        self.status_code = status
        self.headers = {"Content-Type": content_type}
        self.chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("test HTTP error")

    def json(self):
        return self.body

    def iter_content(self, size):
        yield from self.chunks


@pytest.fixture
def remote(tmp_path, monkeypatch):
    management = types.ModuleType("comfy.model_management")
    management.throw_exception_if_processing_interrupted = lambda: None
    comfy = types.ModuleType("comfy")
    comfy.model_management = management
    folders = types.ModuleType("folder_paths")
    folders.get_output_directory = lambda: str(tmp_path)
    for name, module in (("comfy", comfy), ("comfy.model_management", management), ("folder_paths", folders)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(node, "connection_settings", lambda: ("http://dgx.example", "test-key"))
    monkeypatch.setattr(node.time, "sleep", lambda _: None)
    job_id = "video_" + "a" * 32

    class Session:
        def __init__(self):
            self.headers = {}
            self.calls = []
            self.replies = [Response({"status": "queued"}), Response({"status": "completed"}), Response()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, **kwargs):
            self.calls.append((url, kwargs))
            if url.endswith("/files"):
                assert kwargs["files"]["file"][1].read() == b"input-media"
                return Response({"id": "file_" + "b" * 32, "type": "audio"})
            return Response({"id": job_id})

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return self.replies.pop(0)

    session = Session()
    monkeypatch.setattr(node.requests, "Session", lambda: session)
    return session, management, tmp_path / "h3_api" / f"{job_id}.mp4"


def generate(**kwargs):
    return node.H3RemoteVideo().generate("t2va", "test prompt", 5, 768, 42, "16:9", **kwargs)


def test_remote_roundtrip(remote):
    session, _, path = remote
    result = generate(sampling="custom", steps=12)
    assert result["result"] == (str(path),)
    assert result["ui"]["h3_videos"] == [
        {"filename": path.name, "subfolder": "h3_api", "type": "output"}
    ]
    assert "test-key" not in json.dumps(result["ui"])
    assert "dgx.example" not in json.dumps(result["ui"])
    assert node.WEB_DIRECTORY == "./web"
    assert path.read_bytes() == b"test-mp4"
    assert session.headers == {"Authorization": "Bearer test-key"}
    assert session.trust_env is False
    assert all(call[1]["allow_redirects"] is False for call in session.calls)
    payload = session.calls[0][1]["json"]
    assert payload["steps"] == 12 and "preset" not in payload
    assert payload["conditions"] == []
    assert payload["target"]["duration_seconds"] == 5
    assert "test-key" not in json.dumps(payload)


def test_reference_upload(remote, tmp_path):
    session, _, _ = remote
    media = tmp_path / "reference.wav"
    media.write_bytes(b"input-media")
    generate(reference_audio_path=str(media))
    assert session.calls[1][1]["json"]["conditions"] == [
        {"file_id": "file_" + "b" * 32, "type": "audio", "role": "reference"}
    ]


@pytest.mark.parametrize("response", [
    Response(status=302), Response(status=401), Response(content_type="text/html"), Response(chunks=()),
])
def test_bad_download_leaves_no_file(remote, response):
    session, _, path = remote
    session.replies[-1] = response
    with pytest.raises((RuntimeError, requests.HTTPError)):
        generate()
    assert not path.exists()
    assert not path.with_suffix(".partial.mp4").exists()


def test_interrupted_download_cleans_partial(remote):
    session, management, path = remote

    def chunks():
        yield b"first-chunk"
        management.throw_exception_if_processing_interrupted = lambda: (_ for _ in ()).throw(InterruptedError())
        yield b"second-chunk"

    session.replies[-1] = Response(chunks=chunks())
    with pytest.raises(InterruptedError):
        generate()
    assert not path.exists()
    assert not path.with_suffix(".partial.mp4").exists()


def test_remote_failure(remote):
    session, _, path = remote
    session.replies = [Response({"status": "failed", "error": "generation failed"})]
    with pytest.raises(RuntimeError, match="generation failed"):
        generate()
    assert not path.exists()


def test_archive_excludes_local_configuration(tmp_path):
    client = tmp_path / "comfyui_client"
    client.mkdir()
    for name in CLIENT_FILES:
        (client / name).parent.mkdir(parents=True, exist_ok=True)
        (client / name).write_text("public source")
    for name in ("LICENSE", "NOTICE.md"):
        (tmp_path / name).write_text("public license")
    (client / "config.json").write_text("private-key")
    (client / "config.json.bak").write_text("private-key")
    with zipfile.ZipFile(build_archive(tmp_path)) as archive:
        assert set(archive.namelist()) == {
            "h3_api_client/" + name for name in (*CLIENT_FILES, "LICENSE", "NOTICE.md")
        }
        assert all(b"private-key" not in archive.read(name) for name in archive.namelist())
