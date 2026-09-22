"""DGX-only kernel gate. Synthetic numerical sanity is NOT a video quality test."""

import json
import platform


def main():
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("Run only inside the DGX ARM64 benchmark container")
    import torch
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("This numerical gate requires the GB10 GPU (sm_121)")
    torch.manual_seed(42)
    records = []
    with torch.inference_mode():
        for n, k in ((21504, 5376), (28672, 5376), (5376, 14336)):
            x = torch.randn(64, k, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
            qdata, params = TensorWiseINT8Layout.quantize(
                weight,
                is_weight=True,
                per_channel=True,
                convrot=True,
                convrot_groupsize=256,
                stochastic_rounding=0,
            )
            result = torch.ops.comfy_kitchen.int8_linear(
                x,
                qdata,
                params.scale.float(),
                None,
                2,
                True,
                256,
            )
            reference = torch.nn.functional.linear(x, weight).float()
            torch.cuda.synchronize()
            finite = bool(torch.isfinite(result).all())
            error = float((result.float() - reference).norm() / reference.norm().clamp_min(1e-12))
            record = {"shape_mnk": [64, n, k], "finite": finite, "relative_l2": error}
            records.append(record)
            print(json.dumps(record), flush=True)
            if not finite or error > 0.15:
                raise RuntimeError("INT8 kernel sanity failed; do not run the 35-minute benchmark")
            del x, weight, result, reference, qdata, params
            torch.cuda.empty_cache()
    print(json.dumps({"gate": "passed", "torch": torch.__version__, "tests": records}), flush=True)


if __name__ == "__main__":
    main()
