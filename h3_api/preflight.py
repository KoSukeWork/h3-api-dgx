import argparse
import json
import platform
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Check a DGX Spark before installing H3")
    parser.add_argument("--disk", type=Path, default=Path("/srv"))
    parser.add_argument(
        "--runtime", action="store_true", help="Also test the inference venv CUDA/BF16 runtime"
    )
    args = parser.parse_args()
    issues = []
    machine = platform.machine()
    if platform.system() != "Linux" or machine not in ("aarch64", "arm64"):
        issues.append("Expected Linux ARM64 on DGX Spark")
    free = shutil.disk_usage(args.disk).free / 1024**3
    if free < 350:
        issues.append(f"{args.disk}: only {free:.1f} GiB free; need 350 GiB before installation")
    memory = None
    if Path("/proc/meminfo").exists():
        values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        memory = int(values["MemTotal"].split()[0]) / 1024**2
        if memory < 110:
            issues.append(f"Only {memory:.1f} GiB system memory detected")
    gpu = ""
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        gpu = result.stdout.strip()
        if result.returncode or "GB10" not in gpu or len(gpu.splitlines()) != 1:
            issues.append("Expected exactly one GB10; inspect nvidia-smi")
    else:
        issues.append("nvidia-smi is missing; finish NVIDIA system setup first")
    report = {"architecture": machine, "gpu_and_driver": gpu, "memory_GiB": memory, "disk_free_GiB": free}
    if args.runtime:
        if issues:
            report["issues"] = issues
            print(json.dumps(report, indent=2))
            raise SystemExit(1)
        import torch

        report.update(torch=torch.__version__, cuda=torch.version.cuda)
        if torch.version.cuda != "13.0":
            issues.append("Expected the pinned CUDA 13.0 PyTorch build, not a CPU/other CUDA wheel")
        try:
            import torchaudio

            report["torchaudio"] = torchaudio.__version__
            # Detect shared-library/ABI problems in the installed ARM audio stack.
            audio = torchaudio.functional.resample(torch.zeros(1, 3200), 32000, 16000)
            if audio.shape != (1, 1600):
                issues.append("Audio resampling smoke test failed")
        except (ImportError, OSError, RuntimeError) as exc:
            issues.append(f"ARM torchaudio runtime failed: {exc}")
        if not torch.cuda.is_available():
            issues.append("PyTorch cannot access CUDA")
        else:
            report["device_capability"] = torch.cuda.get_device_capability()
            if report["device_capability"] != (12, 1):
                issues.append("Expected GB10 compute capability 12.1")
            try:
                a = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
                if not torch.isfinite(a @ a).all().item():
                    issues.append("BF16 CUDA matmul returned non-finite values")
                q = a.reshape(1, 1, 128, 128)
                y = torch.nn.functional.scaled_dot_product_attention(q, q, q)
                if not torch.isfinite(y).all().item():
                    issues.append("BF16 attention returned non-finite values")
                torch.cuda.synchronize()
            except RuntimeError as exc:
                issues.append(f"CUDA smoke test failed: {exc}")
    report["issues"] = issues
    print(json.dumps(report, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
