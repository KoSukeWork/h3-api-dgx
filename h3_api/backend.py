"""Own exactly one SGLang process tree. Never leave a timed-out generation running."""

import asyncio
import json
import os
import signal
import socket
import time
from pathlib import Path

import httpx

from h3_api.settings import Settings


def linux_group_has_live_members(group_id):
    """Check descendants too; zombies no longer retain a CUDA context."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            stat = (entry / "stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            continue
        fields = stat[stat.rfind(")") + 2 :].split()
        if int(fields[2]) == group_id and fields[0] not in ("Z", "X"):
            return True
    return False


class SGLangBackend:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.process = None
        self.variant = None
        self.state = "idle"
        self.log = None
        self.client = httpx.AsyncClient(base_url=settings.backend_url, timeout=30, trust_env=False)

    @property
    def runtime_dir(self):
        return (self.settings.data_dir / "sglang").resolve()

    def launch_context(self, variant=None):
        """Keep every backend write in the persistent data mount, not the app tree."""
        cache = (self.settings.data_dir / "cache").resolve()
        spill = cache / "host_spill"
        if variant is not None:
            self.validate_comfy_qkv_contract(variant)
            # Never reuse QKV spill files produced by the old, incorrect permutation.
            # Keep old cache files intact for rollback; this namespace is fresh.
            spill = spill / "comfy-bf16-concat-v1"
        directories = [
            self.runtime_dir,
            self.runtime_dir / "inputs",
            self.runtime_dir / "outputs",
            cache / "huggingface",
            cache / "sglang",
            spill,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        for name in ("H3_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "H3_COMFY_BF16_QKV_LAYOUT"):
            env.pop(name, None)
        if variant is not None:
            env["H3_COMFY_BF16_QKV_LAYOUT"] = "concatenated"
        env.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HOME": str(cache / "huggingface"),
                "SGLANG_DIFFUSION_CACHE_ROOT": str(cache / "sglang"),
                # This is a separate upstream setting, not covered by CACHE_ROOT.
                "SGLANG_DIFFUSION_HOST_SPILL_DIR": str(spill),
            }
        )
        return {"cwd": str(self.runtime_dir), "env": env}

    def validate_comfy_qkv_contract(self, variant):
        """Enable the layout override only for our prepared, hash-verified exports.

        This checks the saved SHA verification and current file size/path. It does
        not rehash 66 GB at each startup; rerun prepare if weights are replaced.
        """
        from h3_api.prepare import WEIGHTS, WEIGHTS_REVISION

        manifest = json.loads(
            (self.settings.models_dir / "prepared" / "manifest.json").read_text(encoding="utf-8")
        )
        if variant not in ("fl2va", "ref2va"):
            raise ValueError("Unsupported QKV layout variant")
        _, size, digest, _ = WEIGHTS[variant]
        verified = manifest.get("verified_weights", {}).get(variant, {})
        path = Path(manifest["variants"][variant]["transformer"])
        if (
            manifest.get("weights_revision") != WEIGHTS_REVISION
            or verified.get("sha256") != digest
            or verified.get("bytes") != size
            or not verified.get("path")
            or Path(verified["path"]).resolve() != path.resolve()
            or path.stat().st_size != size
        ):
            raise ValueError("Comfy BF16 QKV layout requires the pinned SHA-verified weights; rerun prepare")

    def command(self, variant):
        manifest_path = self.settings.models_dir / "prepared" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported model manifest; rerun h3-prepare")
        partitions = {"fl2va": "FL2VA", "ref2va": "Ref2VA"}
        if variant not in partitions:
            raise ValueError(f"Unsupported model variant: {variant}")
        config_path = Path(manifest["config_root"]) / partitions[variant]
        index = json.loads((config_path / "model_index.json").read_text(encoding="utf-8"))
        if (
            index.get("_class_name") != "MiniMaxH3Pipeline"
            or not index.get("_diffusers_version")
            or index.get("_minimax_h3", {}).get("partition") != variant
        ):
            raise ValueError(f"Invalid native {variant} pipeline configuration: {config_path}")
        for component in ("transformer", "text_encoder", "video_vae", "audio_vae", "processor", "tokenizer"):
            if not (config_path / component).is_dir():
                raise FileNotFoundError(config_path / component)
        components = manifest["variants"][variant]
        if set(components) != {"transformer", "text_encoder", "video_vae", "audio_vae"}:
            raise ValueError("All four local component weight overrides are required")
        args = [
            self.settings.sglang,
            "serve",
            "--model-path",
            str(config_path),
            # Registry discovery needs the native partition index, not the modular
            # root index. The explicit current subfolder keeps config-only loading
            # local; all weights come from the four overrides below. Do not also
            # pass --model-variant: H3 would append FL2VA/Ref2VA a second time.
            "--model-subfolder",
            ".",
            "--num-gpus",
            "1",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.settings.backend_port),
            "--enable-torch-compile",
            "false",
            "--input-save-path",
            str(self.runtime_dir / "inputs"),
            "--output-path",
            str(self.runtime_dir / "outputs"),
        ]
        for component, path in components.items():
            if not Path(path).is_file():
                raise FileNotFoundError(path)
            args += [f"--component-weights-paths.{component}", path]
        if self.settings.turbo_lora is not None:
            if variant != "fl2va":
                raise ValueError("FL2VA Turbo adapter cannot be applied to Ref2VA")
            lora = self.settings.turbo_lora
            if not lora.is_file():
                raise FileNotFoundError(lora)
            args += [
                "--warmup-mode",
                "off",
                "--lora-path",
                str(lora.parent),
                "--lora-weight-name",
                lora.name,
                "--lora-nickname",
                "h3-turbo-v4",
                "--lora-scale",
                "1.0",
                "--lora-merge-mode",
                "dynamic",
            ]
        return args

    async def ensure(self, variant):
        if (
            self.variant == variant
            and self.process is not None
            and self.process.returncode is None
            and self.state == "ready"
        ):
            return
        await self.stop()
        if self.settings.turbo_lora is not None:
            from h3_api.turbo import verify_lora

            await asyncio.to_thread(verify_lora, self.settings.turbo_lora)
        args = self.command(variant)
        context = self.launch_context(variant)
        # A different service must not be mistaken for our new worker during startup.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", self.settings.backend_port))
        self.state = "loading"
        self.variant = variant
        logs = self.settings.data_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        self.log = (logs / f"sglang-{variant}.log").open("ab", buffering=0)
        log_offset = self.log.tell()
        self.log.write(
            b"\nH3 API: verified Comfy BF16 concatenated QKV; duplicate reorder disabled; spill v1.\n"
        )
        try:
            self.process = await asyncio.create_subprocess_exec(
                *args,
                stdout=self.log,
                stderr=self.log,
                **context,
                start_new_session=(os.name == "posix"),
            )
            deadline = time.monotonic() + self.settings.load_timeout
            while time.monotonic() < deadline:
                if self.process.returncode is not None:
                    raise RuntimeError(f"SGLang exited with {self.process.returncode}; see {logs}")
                try:
                    response = await self.client.get("/health")
                    if response.status_code == 200:
                        if self.settings.turbo_lora is not None:
                            from h3_api.turbo import verify_activation

                            with (logs / f"sglang-{variant}.log").open("rb") as stream:
                                stream.seek(log_offset)
                                startup_log = stream.read().decode("utf-8", errors="replace")
                            verify_activation(startup_log, self.settings.turbo_lora)
                        self.state = "ready"
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(self.settings.poll_interval)
            raise TimeoutError("model_load_timeout")
        except BaseException:
            await self.stop()
            raise

    async def generate(self, payload, destination):
        response = await self.client.post("/v1/videos", json=payload)
        response.raise_for_status()
        backend_id = response.json()["id"]
        # Treat backend IDs as a single URL segment.
        from urllib.parse import quote

        route = "/v1/videos/" + quote(str(backend_id), safe="")
        while True:
            if self.process is None or self.process.returncode is not None:
                raise RuntimeError("backend_exited_during_generation")
            status = await self.client.get(route)
            status.raise_for_status()
            body = status.json()
            if body["status"] == "completed":
                break
            if body["status"] in ("failed", "cancelled", "canceled"):
                raise RuntimeError(f"backend_failed: {str(body.get('error', body['status']))[:1000]}")
            await asyncio.sleep(self.settings.poll_interval)
        async with self.client.stream("GET", route + "/content") as result:
            result.raise_for_status()
            with destination.open("wb") as stream:
                async for chunk in result.aiter_bytes():
                    stream.write(chunk)

    async def stop(self):
        proc = self.process
        if proc is not None:
            if os.name == "posix":
                # Signal the entire session, including children of an already-exited launcher.
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            elif proc.returncode is None:
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), 30)
            except TimeoutError:
                if os.name == "posix":
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    proc.kill()
                await proc.wait()
            if os.name == "posix":
                # A launcher can exit before its CUDA children. Reap the remaining group.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                # GB10 shares one memory pool: wait for all CUDA children before
                # allowing a second full model to start. Fail closed if stuck.
                deadline = time.monotonic() + 10
                while linux_group_has_live_members(proc.pid):
                    if time.monotonic() >= deadline:
                        self.state = "shutdown_failed"
                        raise RuntimeError("Backend children still alive; model switching blocked")
                    await asyncio.sleep(0.1)
        self.process = None
        if self.log:
            self.log.close()
            self.log = None
        self.variant = None
        self.state = "idle"

    async def close(self):
        await self.stop()
        await self.client.aclose()
