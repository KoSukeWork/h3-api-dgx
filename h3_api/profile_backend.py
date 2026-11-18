"""Cold-switch approved recipes through one SGLang process group."""

import asyncio
from dataclasses import replace

from h3_api import ref2va, turbo
from h3_api.backend import SGLangBackend
from h3_api.profiles import LARRY, LIGHTX


class ProfileBackend(SGLangBackend):
    def __init__(self, settings):
        if not settings.turbo_lora or not settings.ref2va_lora:
            raise ValueError("Multi-profile service requires both pinned adapters")
        self.adapters = {LARRY: settings.turbo_lora, LIGHTX: settings.ref2va_lora}
        self.loaded_profile = None
        self.loaded_key = None
        self.selected_profile = None
        # Disable the base class's FL2VA-only automatic adapter; select explicitly below.
        super().__init__(replace(settings, turbo_lora=None, ref2va_lora=None))

    def command(self, variant):
        expected = LIGHTX if variant == "ref2va" else LARRY
        if self.selected_profile != expected or variant not in ("fl2va", "ref2va"):
            raise ValueError("Profile and base-model variant mismatch")
        command = super().command(variant)
        if expected == LIGHTX:
            command = ref2va.route(command, self.settings.data_dir / "config-views" / ref2va.RECIPE_KEY)
        asset = self.adapters[expected]
        command += ["--warmup-mode", "off", "--lora-path", str(asset.parent),
                    "--lora-weight-name", asset.name, "--lora-nickname", expected,
                    "--lora-scale", "1.0", "--lora-merge-mode", "dynamic"]
        if expected == LIGHTX:
            command += ["--lora-alpha", "8"]
        return command

    def launch_context(self, variant=None):
        context = super().launch_context(variant)
        context["env"].pop("H3_REF2VA_REFERENCE_POLICY", None)
        if self.selected_profile == LIGHTX:
            context["env"]["H3_REF2VA_REFERENCE_POLICY"] = "lightx2v-match-v1"
        key = ref2va.RECIPE_KEY if self.selected_profile == LIGHTX else (
            "larry-v4-" + turbo.LORA_SHA256[:12] + "-scale1-qkvconcat-v1"
        )
        cache = self.settings.data_dir / "cache" / "profiles" / key
        for variable, leaf in (("HF_HOME", "hf"), ("SGLANG_DIFFUSION_CACHE_ROOT", "sglang"),
                               ("SGLANG_DIFFUSION_HOST_SPILL_DIR", "spill"),
                               ("TRITON_CACHE_DIR", "triton"), ("TORCH_HOME", "torch"),
                               ("CUDA_CACHE_PATH", "cuda")):
            directory = cache / leaf
            directory.mkdir(parents=True, exist_ok=True)
            context["env"][variable] = str(directory)
        return context

    async def ensure(self, variant, profile=None):
        expected = LIGHTX if variant == "ref2va" else LARRY
        if profile != expected or variant not in ("fl2va", "ref2va"):
            raise ValueError("Profile and base-model variant mismatch")
        key = (variant, profile, ref2va.RECIPE_KEY if profile == LIGHTX else turbo.LORA_SHA256)
        if (self.loaded_key == key and self.state == "ready"
                and self.process is not None and self.process.returncode is None):
            return
        # Wait for every old process-group member before hashing or loading a new recipe.
        await self.stop()
        self.selected_profile = profile
        verifier = ref2va.verify_lora if profile == LIGHTX else turbo.verify_lora
        await asyncio.to_thread(verifier, self.adapters[profile])
        path = self.settings.data_dir / "logs" / f"sglang-{variant}.log"
        offset = path.stat().st_size if path.exists() else 0
        try:
            # Base ensure calls stop again; stop intentionally preserves selected_profile.
            await super().ensure(variant)
            with path.open("rb") as stream:
                stream.seek(offset)
                log = stream.read().decode(errors="replace")
            check = ref2va.verify_active if profile == LIGHTX else turbo.verify_activation
            check(log, self.adapters[profile])
            self.loaded_profile, self.loaded_key = profile, key
        except BaseException:
            await self.stop()
            raise

    async def stop(self):
        await super().stop()
        self.loaded_profile = self.loaded_key = None
