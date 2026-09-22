import os
from dataclasses import dataclass
from pathlib import Path

SGLANG_REVISION = "70b5b03e78612c94f86ac98eb4d2d8d19ceda738"


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    models_dir: Path
    api_key: str
    sglang: str = "/opt/h3-api/inference-venv/bin/sglang"
    backend_port: int = 30010
    load_timeout: float = 2700
    task_timeout: float = 7200
    poll_interval: float = 2
    max_upload_bytes: int = 100 * 1024 * 1024
    max_queue: int = 32
    turbo_lora: Path | None = None

    @property
    def backend_url(self):
        return f"http://127.0.0.1:{self.backend_port}"

    @classmethod
    def from_env(cls):
        key = os.environ.get("H3_API_KEY", "")
        if len(key) < 24:
            raise ValueError("Set H3_API_KEY to a random secret of at least 24 characters")
        return cls(
            data_dir=Path(os.environ.get("H3_DATA_DIR", "/srv/h3/data")),
            models_dir=Path(os.environ.get("H3_MODELS_DIR", "/srv/h3/models")),
            api_key=key,
            sglang=os.environ.get("H3_SGLANG", "/opt/h3-api/inference-venv/bin/sglang"),
            backend_port=int(os.environ.get("H3_BACKEND_PORT", "30010")),
            load_timeout=float(os.environ.get("H3_LOAD_TIMEOUT", "2700")),
            task_timeout=float(os.environ.get("H3_TASK_TIMEOUT", "7200")),
            turbo_lora=Path(os.environ["H3_TURBO_LORA"]) if os.environ.get("H3_TURBO_LORA") else None,
        )
