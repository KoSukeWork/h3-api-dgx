from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Condition(StrictModel):
    type: Literal["image", "video", "audio"]
    file_id: str = Field(pattern=r"^file_[a-f0-9]{32}$")
    role: Literal["keyframe", "reference"] = "reference"
    frame_index: int | None = None

    @model_validator(mode="after")
    def validate_role(self) -> Self:
        if self.role == "keyframe":
            if self.type != "image" or self.frame_index not in (0, -1):
                raise ValueError("Keyframes must be images with frame_index 0 (first) or -1 (last)")
        elif self.frame_index is not None:
            raise ValueError("Reference conditions cannot specify frame_index")
        return self


class Target(StrictModel):
    short_edge: Literal[480, 768] = 480
    aspect_ratio: Literal["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "auto"] = "16:9"
    duration_seconds: float = Field(default=5, ge=4, le=15)


class VideoRequest(StrictModel):
    task: Literal["t2va", "fl2va", "ref2va"]
    prompt: str = Field(min_length=1, max_length=7000)
    conditions: list[Condition] = Field(default_factory=list, max_length=12)
    target: Target = Field(default_factory=Target)
    seed: int = Field(default=42, ge=0, le=2147483647)
    quality: Literal["lossless"] = "lossless"
    preset: Literal["fast", "quality"] | None = None
    steps: int | None = Field(
        default=None,
        ge=1,
        le=49,
        strict=True,
        description="Actual denoiser evaluations; not sigma-point count. Turbo only.",
    )

    def sampling(self, turbo: bool) -> dict:
        if turbo:
            if self.variant != "fl2va":
                raise ValueError("This Turbo LoRA supports t2va/fl2va only; ref2va needs a separate recipe")
            steps = self.steps if self.steps is not None else (4 if self.preset == "fast" else 8)
        else:
            if self.preset is not None or self.steps is not None:
                raise ValueError("preset/steps require a Turbo-enabled service")
            steps = 49
        return {"mode": "turbo" if turbo else "base", "steps": steps, "num_inference_steps": steps + 1}

    @property
    def variant(self) -> str:
        return "ref2va" if self.task == "ref2va" else "fl2va"

    @model_validator(mode="after")
    def validate_task(self) -> Self:
        if self.preset is not None and self.steps is not None:
            raise ValueError("Specify either preset or steps, not both")
        if not self.prompt.strip():
            raise ValueError("Prompt cannot be blank")
        keys = [c for c in self.conditions if c.role == "keyframe"]
        refs = [c for c in self.conditions if c.role == "reference"]
        if len(keys) > 2 or len({c.frame_index for c in keys}) != len(keys):
            raise ValueError("At most one first and one last keyframe are allowed")
        if self.task == "t2va" and self.conditions:
            raise ValueError("t2va takes no conditions")
        if self.task == "t2va" and self.target.aspect_ratio == "auto":
            raise ValueError("t2va requires an explicit aspect_ratio")
        if self.task == "fl2va" and (not keys or refs):
            raise ValueError("fl2va requires first/last keyframes, not references")
        if self.task == "ref2va":
            if not refs or not any(c.type in ("image", "video") for c in refs):
                raise ValueError("ref2va requires at least one image/video reference")
            for kind, limit in (("image", 9), ("video", 3), ("audio", 3)):
                if sum(c.type == kind for c in refs) > limit:
                    raise ValueError(f"At most {limit} {kind} references are allowed")
        return self
