"""DGX-only preparation and read-only cleanup inventory. No deletions or service starts."""

import argparse
import json
import os
import platform
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE))

from h3_api.turbo import LORA_FILENAME, verify_lora

MODELS = Path("/srv/h3/models")
DATA = Path("/srv/h3/service-data")
LORA_SOURCE = Path("/srv/h3/data/benchmark/assets/turbo") / LORA_FILENAME
BASE_IMAGE = "h3-bench:gb10-turbo"


def host_guard():
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("Run only on the DGX ARM64 host, not Windows/x86")
    if os.geteuid() != 0:
        raise RuntimeError("Run with sudo")


def old_services_stopped():
    for project in ("h3", "h3-bench", "h3-service"):
        active = subprocess.check_output(
            [
                "docker",
                "ps",
                "--filter",
                f"label=com.docker.compose.project={project}",
                "--format",
                "{{.Names}}",
            ],
            text=True,
        ).strip()
        if active:
            raise RuntimeError(f"Stop these H3 containers before preparing: {active}")
    gpu_jobs = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    if gpu_jobs:
        raise RuntimeError(f"GPU is busy; do not prepare another H3 deployment: {gpu_jobs}")


def install_lora(source, destination):
    if destination.is_symlink():
        raise RuntimeError("Refusing a symlink as the production LoRA target")
    if destination.exists():
        verify_lora(destination)
        return
    verify_lora(source)
    # Verify a staging copy before publishing it. link() refuses to replace any
    # destination created concurrently; the original benchmark asset is retained.
    fd, temporary_name = tempfile.mkstemp(prefix=".h3-lora-", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as target, source.open("rb") as origin:
            shutil.copyfileobj(origin, target, 16 * 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        verify_lora(temporary)
        temporary.chmod(0o644)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)  # Only our own uniquely named staging file.


def write_env(path):
    if path.is_symlink():
        raise RuntimeError("Refusing a symlink as .env")
    if path.exists():
        print("Preserving existing .env (API key is never printed).")
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(f"H3_API_KEY={secrets.token_urlsafe(32)}\nH3_BIND_IP=127.0.0.1\nH3_PORT=8000\n")


def prepare():
    old_services_stopped()
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", BASE_IMAGE], text=True))[0]
    if image["Architecture"] != "arm64":
        raise RuntimeError("The successful ARM64 Turbo benchmark image is required")
    manifest = MODELS / "prepared" / "manifest.json"
    if not manifest.is_file():
        raise RuntimeError(f"Missing prepared model manifest: {manifest}; keep the original models directory")
    if DATA.is_symlink():
        raise RuntimeError("Production data directory must not be a symlink")
    if DATA.exists():
        if DATA.stat().st_uid != 10001:
            raise RuntimeError("Existing service-data has a different owner; do not overwrite it")
        if any(DATA.iterdir()):
            raise RuntimeError(
                "service-data is not empty; preserve it and do not rerun first-time preparation"
            )
    lora_dir = MODELS / "loras"
    if lora_dir.is_symlink():
        raise RuntimeError("Production LoRA directory must not be a symlink")
    lora_dir.mkdir(mode=0o755, exist_ok=True)
    print("Verifying and copying the existing Turbo adapter; no model downloads.", flush=True)
    install_lora(LORA_SOURCE, lora_dir / LORA_FILENAME)
    DATA.mkdir(mode=0o750, exist_ok=True)
    os.chown(DATA, 10001, 10001)
    write_env(PACKAGE / ".env")
    subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            str(PACKAGE / ".env"),
            "-f",
            str(PACKAGE / "production" / "compose.yaml"),
            "config",
            "--quiet",
        ],
        check=True,
    )
    print(
        json.dumps(
            {
                "base_image": BASE_IMAGE,
                "image_id": image["Id"],
                "data_directory": str(DATA),
                "lora": str(lora_dir / LORA_FILENAME),
                "status": "prepared; no containers started or files deleted",
            },
            indent=2,
        )
    )


def inventory_paths(legacy_install_dir=None):
    paths = [
        MODELS,
        DATA,
        Path("/srv/h3/data/cache"),
        Path("/srv/h3/data/sglang"),
        Path("/srv/h3/data/logs"),
        Path("/srv/h3/data/acceptance"),
        Path("/srv/h3/data/uploads"),
        Path("/srv/h3/data/outputs"),
        Path("/srv/h3/data/benchmark/assets"),
        Path("/srv/h3/data/benchmark/cache"),
        Path("/srv/h3/data/benchmark/runs"),
        PACKAGE,
    ]
    if legacy_install_dir is not None:
        legacy_install_dir = Path(legacy_install_dir)
        if not legacy_install_dir.is_absolute():
            raise ValueError("--legacy-install-dir must be an absolute path")
        paths.append(legacy_install_dir)
    return list(dict.fromkeys(paths))


def inventory(legacy_install_dir=None):
    paths = inventory_paths(legacy_install_dir)
    sizes = {}
    for path in paths:
        if path.exists():
            sizes[str(path)] = subprocess.check_output(["du", "-sh", "--", str(path)], text=True).strip()
    print(json.dumps({"directory_sizes": sizes, "action": "read-only; nothing deleted"}, indent=2))
    subprocess.run(
        ["docker", "ps", "-a", "--format", "table {{.Names}}\t{{.Image}}\t{{.Status}}"], check=True
    )
    subprocess.run(["docker", "system", "df"], check=True)
    print("KEEP models, LoRA, certificates, all generated results, and successful runtime images.")
    print("Review this inventory before removing named caches. Do NOT use docker system prune -a.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "inventory"))
    parser.add_argument(
        "--legacy-install-dir",
        type=Path,
        help="Optional absolute old installation path to inspect; inventory only",
    )
    args = parser.parse_args()
    if args.legacy_install_dir is not None:
        if args.action != "inventory":
            parser.error("--legacy-install-dir is only valid for inventory")
        if not args.legacy_install_dir.is_absolute():
            parser.error("--legacy-install-dir must be an absolute path")
    host_guard()
    if args.action == "prepare":
        prepare()
    else:
        inventory(args.legacy_install_dir)


if __name__ == "__main__":
    main()
