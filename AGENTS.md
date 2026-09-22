# Repository working rules

- Read README.md and the relevant production/bootstrap documentation before changing deployment behavior.
- Target: DGX Spark GB10, Linux aarch64, sm_121. Never run GPU inference, ARM image builds or GPU package installation on the Windows/x86 development machine. CPU-only unit tests and Compose config validation are allowed.
- Default production deployment is non-quantized BF16 Transformer + Larry Turbo v4, not local ComfyUI inference. Quantization tooling under benchmark is optional historical comparison only.
- Preserve the pinned QKV-layout and LoRA/MXFP8 compatibility fixes. Do not bypass their source/hash checks or apply BF16 assumptions to arbitrary checkpoints.
- presets fast/quality map to 4/8 actual evaluations; custom steps are actual evaluations, while the internal SGLang request uses steps + 1. Step changes must not reload the same model.
- Keep one GPU model worker. Current Turbo LoRA is FL2VA-specific; do not silently apply it to Ref2VA.
- Never commit API keys, HF tokens, local CA certificates, weights, generated videos, caches or image archives. Package scripts must retain their explicit allowlist.
- Cleanup must inspect exact targets first and preserve models/results/certificates. Do not delete broad directories or run global Docker prune commands.
- Do not call a new deployment GPU-validated based only on CPU tests or the saved historical benchmark.
- Test with `uv sync --extra test --frozen`, then `uv run pytest -q` and `uv run ruff check .`.
- In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), use CodeGraph before grep/find/file reads for code discovery; without that directory, skip CodeGraph.
